"""Utilities for querying National Flood Data API for a region."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import sys

import json
import geopandas as gpd
import mercantile
import mapbox_vector_tile
from shapely.geometry import shape

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.settings import get_settings

API_URL = "https://api.nationalflooddata.com/v3/data"
OUTPUT_PATH = Path("backend/data/processed/flood/floodzones_sfla.parquet")

settings = get_settings()


class FloodFetcherError(RuntimeError):
    """Raised when the flood data API returns an error."""


def _generate_grid(bbox: str, lat_step: float = 0.1, lon_step: float = 0.1) -> Iterable[Tuple[float, float]]:
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError as exc:
        raise ValueError(
            "FEMA_BBOX must contain comma-separated min_lon,min_lat,max_lon,max_lat values"
        ) from exc

    lat = min_lat
    while lat <= max_lat + 1e-9:
        lon = min_lon
        while lon <= max_lon + 1e-9:
            yield round(lat, 6), round(lon, 6)
            lon += lon_step
        lat += lat_step


def tile_to_geojson(tile_path: Path) -> gpd.GeoDataFrame:
    # parse Z/X/Y from filename tile_Z_X_Y.mvt
    parts = tile_path.stem.split("_")
    z, x, y = map(int, parts[1:4])
    raw = tile_path.read_bytes()
    decoded = mapbox_vector_tile.decode(raw)

    features = []
    for layer_name, layer in decoded.items():
        for feature in layer["features"]:
            geom = shape(feature["geometry"])
            features.append({
                "layer": layer_name,
                **feature["properties"],
                "geometry": geom
            })

    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:3857")
    # optionally reproject to WGS84
    return gdf.to_crs("EPSG:4326")


def _request(payload: Dict[str, str]) -> Dict:
    headers = {"x-api-key": settings.fema_api_key}
    response = requests.get(API_URL, headers=headers, params=payload, timeout=60)
    if response.status_code == 403:
        raise FloodFetcherError("National Flood Data API returned 403 Forbidden. Verify API key and account access.")
    if response.status_code == 404:
        raise FloodFetcherError("National Flood Data API returned 404; verify URL and parameters.")
    response.raise_for_status()
    return response.json()


def fetch_point(lat: float, lng: float, *, include_elevation: bool = False, loma: bool = False) -> Dict:
    payload = {
        "lat": f"{lat}",
        "lng": f"{lng}",
        "searchtype": "addresscoord",
        "loma": str(loma).lower(),
        "elevation": str(include_elevation).lower(),
    }
    return _request(payload)


def fetch_address(address: str, *, include_elevation: bool = False, loma: bool = False) -> Dict:
    payload = {
        "address": address,
        "searchtype": "addressparcel",
        "loma": str(loma).lower(),
        "elevation": str(include_elevation).lower(),
    }
    return _request(payload)


def _flatten_response(lat: Optional[float], lng: Optional[float], payload: Dict) -> List[Dict]:
    if not isinstance(payload, dict):
        return [{"lat": lat, "lng": lng, "raw": payload}]

    data = payload.get("data")
    items = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
    if not items:
        record = {"lat": lat, "lng": lng}
        record.update({k: v for k, v in payload.items() if k != "data"})
        return [record]

    records: List[Dict] = []
    for item in items:
        if isinstance(item, dict):
            record = {"lat": lat, "lng": lng}
            record.update(item)
            records.append(record)
    return records


def fetch_bbox(bbox: str, lat_step: float = 0.1, lon_step: float = 0.1) -> pd.DataFrame:
    records: List[Dict] = []
    for lat, lng in _generate_grid(bbox, lat_step=lat_step, lon_step=lon_step):
        payload = fetch_point(lat, lng)
        records.extend(_flatten_response(lat, lng, payload))
    if not records:
        return pd.DataFrame(columns=["lat", "lng"])
    return pd.json_normalize(records)


def save_floodzones(lat_step: float = 0.1, lon_step: float = 0.1) -> Path:
    df = fetch_bbox(settings.fema_bbox, lat_step=lat_step, lon_step=lon_step)
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} flood records to {output_path}")
    return output_path


if __name__ == "__main__":
    save_floodzones()
