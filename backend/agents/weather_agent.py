from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState

from backend.agents.base_agent import AgentConfig, BaseAgent
from backend.services import weather_fetcher


@dataclass(slots=True)
class CachedWeather:
    features: List[Dict[str, Any]]


class WeatherAgentConfig(AgentConfig):
    """Configuration specific to the weather agent."""

    default_lookback_hours: int = 24
    default_max_features: int = 1000


class IngestWeatherPayload(BaseModel):
    """Command payload for ingesting storm signals.

    area_bbox should be [min_lon, min_lat, max_lon, max_lat]. If omitted, the agent
    will use a default (not set here) and will error unless configured otherwise.
    """

    area_bbox: Optional[List[float]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    include_alerts: bool = True
    include_tracks: bool = True
    include_geojson: bool = True
    max_features: Optional[int] = None


class WeatherAgent(BaseAgent):
    """Agent that ingests live storm signals from NWS/NOAA and produces a
    WeatherContext payload with GeoJSON features and normalized metrics.
    """

    def __init__(self, config: WeatherAgentConfig) -> None:
        super().__init__(config)
        self._cache: Dict[str, CachedWeather] = {}

    @property
    def weather_config(self) -> WeatherAgentConfig:
        return self.config  # type: ignore[return-value]

    @classmethod
    def get_agent_card(cls) -> AgentCard:
        return AgentCard(
            name="weather-agent",
            version="0.1.0",
            description=(
                "Ingests live storm signals (alerts & tracks) from NWS/NOAA APIs and "
                "emits GeoJSON features and normalized storm metrics."
            ),
            url="https://outagent.local/a2a/weather",
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="weather.ingest",
                    name="Ingest storm signals",
                    description=(
                        "Fetch active weather alerts and storm track observations for a bbox "
                        "and time window, normalize metrics, and return a WeatherContext."
                    ),
                    tags=["weather", "noaa", "nws", "ingest"],
                ),
            ],
        )

    async def bootstrap(self, ctx) -> None:
        self.logger.info("weather_agent_bootstrap")

    async def shutdown(self, ctx) -> None:
        self.logger.info("weather_agent_shutdown")
        self._cache.clear()

    @BaseAgent.message_handler("weather.ingest", payload_model=IngestWeatherPayload)
    async def handle_weather_ingest(self, ctx, event_queue, payload: IngestWeatherPayload, message) -> None:
        cfg = self.weather_config
        if not payload.area_bbox:
            await self.send_error(event_queue, ctx, "area_bbox is required for weather.ingest")
            return

        start, end = self._resolve_window(payload)
        max_features = payload.max_features or cfg.default_max_features

        await self.send_status_update(
            event_queue,
            ctx,
            state=TaskState.working,
            status_message=f"Ingesting weather signals for bbox={payload.area_bbox} from {start} to {end}",
        )

        alerts: List[Dict[str, Any]] = []
        tracks: List[Dict[str, Any]] = []

        # Fetch alerts
        if payload.include_alerts:
            try:
                if hasattr(weather_fetcher, "fetch_alerts"):
                    alerts = await weather_fetcher.fetch_alerts(
                        client=self.http_client,
                        bbox=payload.area_bbox,
                        start=start,
                        end=end,
                        max_results=max_features,
                    )
                else:
                    raise AttributeError("weather_fetcher.fetch_alerts is not implemented")
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("fetch_alerts_failed", error=str(exc))
                await self.send_error(event_queue, ctx, f"Failed to fetch alerts: {exc}")
                return

        # Fetch tracks
        if payload.include_tracks:
            try:
                if hasattr(weather_fetcher, "fetch_tracks"):
                    tracks = await weather_fetcher.fetch_tracks(
                        client=self.http_client,
                        bbox=payload.area_bbox,
                        start=start,
                        end=end,
                        max_results=max_features,
                    )
                else:
                    raise AttributeError("weather_fetcher.fetch_tracks is not implemented")
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("fetch_tracks_failed", error=str(exc))
                await self.send_error(event_queue, ctx, f"Failed to fetch tracks: {exc}")
                return

        # Convert to GeoJSON features
        features: List[Dict[str, Any]] = []
        for a in alerts:
            feat = _alert_to_feature(a)
            features.append(feat)
        for t in tracks:
            feat = _track_to_feature(t)
            features.append(feat)

        # Metrics: compute max_wind and rain_24h and a normalized storm_severity [0-1]
        max_wind = _compute_max_wind(tracks)
        rain_24h = _compute_rain_24h(alerts)
        storm_severity = _normalize_severity(max_wind=max_wind, rain_24h=rain_24h)

        weather_context = {
            "geojson": {"type": "FeatureCollection", "features": features},
            "metrics": {
                "storm_severity": storm_severity,
                "max_wind": max_wind,
                "rain_24h": rain_24h,
            },
        }

        # Cache a lightweight representation keyed by context/task
        try:
            task_key = f"{self._resolve_context_id(ctx)}:{self._resolve_task_id(ctx)}"
            self._cache[task_key] = CachedWeather(features)
        except Exception:
            # best-effort caching; don't fail the request if resolving ids fails
            pass

        await self.send_data_response(
            event_queue,
            ctx,
            data=weather_context,
            text=f"Ingested {len(features)} weather features; severity={storm_severity}",
        )

    def _resolve_window(self, payload: IngestWeatherPayload) -> Tuple[datetime, datetime]:
        end = payload.end_time or datetime.utcnow()
        start = payload.start_time or (end - timedelta(hours=self.weather_config.default_lookback_hours))
        return start, end


def _alert_to_feature(alert: Dict[str, Any]) -> Dict[str, Any]:
    # Attempt to produce a GeoJSON Feature from an alert dict. This is defensive/flexible
    geom = alert.get("geometry") or alert.get("bbox") or None
    properties = {k: v for k, v in alert.items() if k != "geometry" and k != "bbox"}
    feature: Dict[str, Any] = {
        "type": "Feature",
        "properties": properties,
        "geometry": None,
    }
    if isinstance(geom, dict):
        feature["geometry"] = geom
    elif isinstance(geom, (list, tuple)) and len(geom) == 4:
        # bbox -> polygon
        min_lon, min_lat, max_lon, max_lat = geom
        feature["geometry"] = {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        }
    else:
        feature["geometry"] = None
    return feature


def _track_to_feature(track: Dict[str, Any]) -> Dict[str, Any]:
    # Convert a track observation (sequence of points) to a LineString feature.
    points = []
    coords = track.get("coords") or track.get("points") or track.get("geometry")
    if isinstance(coords, list):
        # coords may be list of [lon, lat] pairs or list of dicts with lon/lat
        for p in coords:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append([float(p[0]), float(p[1])])
            elif isinstance(p, dict) and "lon" in p and "lat" in p:
                points.append([float(p["lon"]), float(p["lat"])])
    properties = {k: v for k, v in track.items() if k not in ("coords", "points", "geometry")}
    feature: Dict[str, Any] = {
        "type": "Feature",
        "properties": properties,
        "geometry": None,
    }
    if points:
        feature["geometry"] = {"type": "LineString", "coordinates": points}
    else:
        feature["geometry"] = None
    return feature


def _compute_max_wind(tracks: List[Dict[str, Any]]) -> float:
    max_wind = 0.0
    for t in tracks:
        # look for a max wind marker in track properties or in points
        props = t.get("properties") or t
        if isinstance(props, dict):
            for key in ("max_wind", "wind_mph", "wind_kts", "wind"):
                val = props.get(key)
                if val is None:
                    continue
                try:
                    w = float(val)
                    # if value is in kts, convert roughly to mph if key contains 'kts'
                    if "kts" in str(key).lower() and w > 0:
                        w = w * 1.15078
                    if w > max_wind:
                        max_wind = w
                except Exception:
                    continue
    return round(max_wind, 2)


def _compute_rain_24h(alerts: List[Dict[str, Any]]) -> float:
    # Attempt to extract an estimated 24-hour rainfall from alerts. This is heuristic.
    total = 0.0
    count = 0
    for a in alerts:
        props = a.get("properties") or a
        if isinstance(props, dict):
            for key in ("rain_24h", "expected_precip_in", "precip_24h", "qpf"):
                val = props.get(key)
                if val is None:
                    continue
                try:
                    r = float(val)
                    total += r
                    count += 1
                    break
                except Exception:
                    continue
    if count == 0:
        return 0.0
    # Return average observed/estimated rain over matched alerts
    return round(total / max(count, 1), 2)


def _normalize_severity(max_wind: float, rain_24h: float) -> float:
    # Normalize to [0,1] using simple heuristics:
    # - winds: 0 -> 0, 150+ mph -> 1.0
    # - rain: 0 -> 0, 10+ in -> 1.0
    wind_score = min(max_wind / 150.0, 1.0)
    rain_score = min(rain_24h / 10.0, 1.0)
    severity = (wind_score * 0.6) + (rain_score * 0.4)
    return round(min(max(severity, 0.0), 1.0), 3)
