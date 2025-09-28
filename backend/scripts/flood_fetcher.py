from pathlib import Path
from functools import partial
import math
import sys
from typing import Iterable, List, Tuple, Dict

import geopandas as gpd
import mercantile
import mapbox_vector_tile
import requests
from shapely.geometry import shape
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.settings import get_settings

API_TILES_URL = "https://api.nationalflooddata.com/v3/tiles/flood-vector/{z}/{x}/{y}.mvt"
OUTPUT_PATH = Path("backend/data/processed/flood/floodzones_sfla.geojson")

settings = get_settings()


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lon = ((lon + 180.0) % 360.0) - 180.0
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _generate_tile_range(bbox: str, zoom: int) -> Iterable[Tuple[int, int]]:
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError as exc:
        raise ValueError(
            "FEMA_BBOX must contain comma-separated min_lon,min_lat,max_lon,max_lat values"
        ) from exc

    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat

    x_min, y_max = _latlon_to_tile(min_lat, min_lon, zoom)
    x_max, y_min = _latlon_to_tile(max_lat, max_lon, zoom)

    x_start = int(math.floor(min(x_min, x_max)))
    x_end = int(math.floor(max(x_min, x_max)))
    y_start = int(math.floor(min(y_min, y_max)))
    y_end = int(math.floor(max(y_min, y_max)))

    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            yield x, y


def download_vector_tile(z: int, x: int, y: int) -> bytes:
    headers = {"x-api-key": settings.fema_api_key}
    url = API_TILES_URL.format(z=z, x=x, y=y)
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.content


def _decode_tile(z: int, x: int, y: int, tile_bytes: bytes) -> List[Dict]:
    decoded = mapbox_vector_tile.decode(tile_bytes)
    bounds = mercantile.bounds(x, y, z)
    west, south, east, north = bounds

    features: List[Dict] = []

    for layer_name, layer in decoded.items():
        extent = layer.get("extent", 4096)
        scale_x = (east - west) / extent
        scale_y = (north - south) / extent

        def _project(coord_x: float, coord_y: float, coord_z: float | None = None) -> tuple[float, float]:
            lon = west + coord_x * scale_x
            lat = north - coord_y * scale_y
            return lon, lat

        projector = partial(_project)

        for feature in layer.get("features", []):
            geometry = shape(feature["geometry"])
            geometry = transform(projector, geometry)
            record = {
                "tile_z": z,
                "tile_x": x,
                "tile_y": y,
                "layer": layer_name,
                **feature.get("properties", {}),
                "geometry": geometry,
            }
            features.append(record)

    return features


def save_floodzones(zoom: int = 13) -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    for x, y in _generate_tile_range(settings.fema_bbox, zoom):
        tile_bytes = download_vector_tile(zoom, x, y)
        records.extend(_decode_tile(zoom, x, y, tile_bytes))
        print(f"Decoded tile z={zoom}, x={x}, y={y}")

    if not records:
        print("No records retrieved from flood tiles.")
        return OUTPUT_PATH

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved {len(gdf)} flood features to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    save_floodzones()
