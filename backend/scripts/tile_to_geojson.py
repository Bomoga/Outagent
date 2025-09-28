from pathlib import Path
import json

import geopandas as gpd
import mercantile
import mapbox_vector_tile
from shapely.geometry import shape

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

tile = Path("backend/data/processed/flood/tile_13_2234_3493.mvt")
gdf = tile_to_geojson(tile)
gdf.to_file(tile.with_suffix(".geojson"), driver="GeoJSON")