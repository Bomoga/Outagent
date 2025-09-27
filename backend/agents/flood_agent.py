from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState

from backend.agents.base_agent import AgentConfig, BaseAgent
from backend.services import flood_fetcher


@dataclass(slots=True)
class FloodCache:
    results: Dict[str, Any]


class FloodAgentConfig(AgentConfig):
    """Configuration for flood agent."""

    use_nfhl_wms: bool = True
    nfhl_wms_url: Optional[str] = None
    shapely_fallback: bool = True


class FloodRequestPayload(BaseModel):
    """Inputs for flood exposure assessment.

    - weather_context: dict with geojson features + metrics (from WeatherAgent)
    - asset_points: list of {"id": str, "lon": float, "lat": float}
    """

    weather_context: Dict[str, Any]
    asset_points: List[Dict[str, Any]]


class FloodAgent(BaseAgent):
    """Agent that assesses flood and storm surge exposure for assets."""

    def __init__(self, config: FloodAgentConfig) -> None:
        super().__init__(config)
        self._cache: Dict[str, FloodCache] = {}

    @property
    def flood_config(self) -> FloodAgentConfig:
        return self.config  # type: ignore[return-value]

    @classmethod
    def get_agent_card(cls) -> AgentCard:
        return AgentCard(
            name="flood-agent",
            version="0.1.0",
            description=(
                "Assess flood and surge exposure for asset points using FEMA NFHL "
                "or local geometry calculations (shapely fallback)."
            ),
            url="https://outagent.local/a2a/flood",
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            capabilities=AgentCapabilities(streaming=False),
            skills=[
                AgentSkill(
                    id="flood.assess",
                    name="Assess flood exposure",
                    description="Determine if assets are in flood zones, distance to flood, and surge flags.",
                    tags=["flood", "nfhl", "surge"],
                ),
            ],
        )

    async def bootstrap(self, ctx) -> None:
        self.logger.info("flood_agent_bootstrap")

    async def shutdown(self, ctx) -> None:
        self.logger.info("flood_agent_shutdown")
        self._cache.clear()

    @BaseAgent.message_handler("flood.assess", payload_model=FloodRequestPayload)
    async def handle_flood_assess(self, ctx, event_queue, payload: FloodRequestPayload, message) -> None:
        cfg = self.flood_config
        assets = payload.asset_points or []
        if not assets:
            await self.send_error(event_queue, ctx, "asset_points is required and cannot be empty")
            return

        await self.send_status_update(
            event_queue,
            ctx,
            state=TaskState.working,
            status_message=f"Assessing flood exposure for {len(assets)} assets",
        )

        # Try NFHL service first if configured
        results: List[Dict[str, Any]] = []
        use_nfhl = cfg.use_nfhl_wms and hasattr(flood_fetcher, "query_nfhl")
        if use_nfhl:
            try:
                results = await flood_fetcher.query_nfhl(
                    client=self.http_client,
                    assets=assets,
                    wms_url=cfg.nfhl_wms_url,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("nfhl_query_failed", error=str(exc))
                # fallback to shapely if allowed
                if not cfg.shapely_fallback:
                    await self.send_error(event_queue, ctx, f"NFHL query failed: {exc}")
                    return

        # If we don't have results from NFHL, and shapely fallback is allowed, do local geometry
        if not results and cfg.shapely_fallback and hasattr(flood_fetcher, "shapely_assess"):
            try:
                results = await flood_fetcher.shapely_assess(
                    assets=assets,
                    weather_context=payload.weather_context,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("shapely_assess_failed", error=str(exc))
                await self.send_error(event_queue, ctx, f"Local flood assessment failed: {exc}")
                return

        # If no external tools available, try to perform a minimal assessment using shapely if installed locally
        if not results:
            try:
                # attempt local computation without a service helper
                results = _local_minimal_assess(assets, payload.weather_context)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("local_assess_failed", error=str(exc))
                await self.send_error(event_queue, ctx, f"Unable to assess flood exposure: {exc}")
                return

        # Results should be list of per-asset dicts with keys: id, in_flood_zone (bool), flood_distance_m (float), surge_zone_flag (bool)
        flood_context = {"assets": results}

        # Cache best-effort
        try:
            task_key = f"{self._resolve_context_id(ctx)}:{self._resolve_task_id(ctx)}"
            self._cache[task_key] = FloodCache({a["id"]: a for a in results})
        except Exception:
            pass

        await self.send_data_response(
            event_queue,
            ctx,
            data={"flood_context": flood_context},
            text=f"Assessed flood exposure for {len(results)} assets",
        )


def _local_minimal_assess(assets: List[Dict[str, Any]], weather_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Perform a minimal, best-effort flood assessment using only provided weather_context geometries.

    This function prefers shapely if available; otherwise it falls back to simple bbox checks.
    """
    try:
        from shapely.geometry import shape, Point
        from shapely.ops import nearest_points
    except Exception:
        # shapely not available; use crude bbox checks with linear distance in degrees (approx meters conversion)
        return _minimal_bbox_assess(assets, weather_context)

    features = []
    geo = weather_context.get("geojson") or {}
    for feat in geo.get("features", []) if isinstance(geo, dict) else []:
        try:
            features.append(shape(feat.get("geometry")))
        except Exception:
            continue

    results: List[Dict[str, Any]] = []
    for a in assets:
        aid = a.get("id")
        lon = float(a.get("lon"))
        lat = float(a.get("lat"))
        pt = Point(lon, lat)
        in_zone = False
        min_dist_m = float("inf")
        for geom in features:
            try:
                if geom.contains(pt):
                    in_zone = True
                    min_dist_m = 0.0
                    break
                # nearest_points returns (pt_on_geom, pt)
                p_geom, p_pt = nearest_points(geom, pt)
                # shapely distance is in coordinate units (degrees) — convert approx deg->meters at equator
                deg_dist = pt.distance(p_geom)
                meters = deg_dist * 111320.0
                if meters < min_dist_m:
                    min_dist_m = meters
            except Exception:
                continue

        surge_flag = False
        # naive surge flag: if any weather_context metric storm_severity > 0.6, flag true
        metrics = weather_context.get("metrics") or {}
        if metrics.get("storm_severity", 0.0) > 0.6:
            surge_flag = True

        results.append(
            {
                "id": aid,
                "in_flood_zone": bool(in_zone),
                "flood_distance_m": 0.0 if in_zone else round(min_dist_m, 2) if min_dist_m != float("inf") else None,
                "surge_zone_flag": bool(surge_flag),
            }
        )

    return results


def _minimal_bbox_assess(assets: List[Dict[str, Any]], weather_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    geo = weather_context.get("geojson") or {}
    bboxes: List[Tuple[float, float, float, float]] = []
    for feat in geo.get("features", []) if isinstance(geo, dict) else []:
        geom = feat.get("geometry")
        if not geom:
            continue
        # attempt to extract bbox
        if geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [])
            if coords and coords[0]:
                lons = [p[0] for p in coords[0]]
                lats = [p[1] for p in coords[0]]
                bboxes.append((min(lons), min(lats), max(lons), max(lats)))
        elif geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if coords and len(coords) >= 2:
                lons = coords[0]
                lats = coords[1]
                bboxes.append((lons, lats, lons, lats))

    for a in assets:
        aid = a.get("id")
        lon = float(a.get("lon"))
        lat = float(a.get("lat"))
        in_zone = False
        min_deg_dist = float("inf")
        for (min_lon, min_lat, max_lon, max_lat) in bboxes:
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                in_zone = True
                min_deg_dist = 0.0
                break
            # compute simple degree distance to bbox center
            center_lon = (min_lon + max_lon) / 2.0
            center_lat = (min_lat + max_lat) / 2.0
            deg_dist = ((lon - center_lon) ** 2 + (lat - center_lat) ** 2) ** 0.5
            if deg_dist < min_deg_dist:
                min_deg_dist = deg_dist

        meters = None if min_deg_dist == float("inf") else round(min_deg_dist * 111320.0, 2)
        metrics = weather_context.get("metrics") or {}
        surge_flag = bool(metrics.get("storm_severity", 0.0) > 0.6)

        results.append(
            {
                "id": aid,
                "in_flood_zone": bool(in_zone),
                "flood_distance_m": 0.0 if in_zone else meters,
                "surge_zone_flag": surge_flag,
            }
        )

    return results
