import os
import json
import math
import time
import argparse
from datetime import datetime
from pathlib import Path

import requests

NFHL_LAYER_URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

DEFAULT_FIELDS = ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "STATIC_BFE", "V_DATUM", "FLD_AR_ID"]

def _safe_filename(s: str) -> str:
    return s.replace(" ", "_").replace(",", "_").replace(":", "_")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _tile_bounds(min_lat, min_lon, max_lat, max_lon, rows, cols):
    """Yield (r, c, (tile_min_lat, tile_min_lon, tile_max_lat, tile_max_lon))"""
    dlat = (max_lat - min_lat) / rows
    dlon = (max_lon - min_lon) / cols
    for r in range(rows):
        for c in range(cols):
            tmin_lat = min_lat + r * dlat
            tmax_lat = tmin_lat + dlat
            tmin_lon = min_lon + c * dlon
            tmax_lon = tmin_lon + dlon
            yield r, c, (tmin_lat, tmin_lon, tmax_lat, tmax_lon)

def _fema_query_bbox(min_lat, min_lon, max_lat, max_lon, fields, page_size=2000, sleep=0.25, timeout=60):
    out_fields = ",".join(fields) if isinstance(fields, (list, tuple)) else (fields or "*")

    # ArcGIS uses envelope geometry: xmin,ymin,xmax,ymax (lon,lat order) in WGS84
    geometry = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    result_offset = 0
    all_features = []

    while True:
        params = {
            "where": "1=1",
            "geometry": geometry,
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "outFields": out_fields,
            "outSR": 4326,
            "returnGeometry": "true",
            "spatialRel": "esriSpatialRelIntersects",
            "f": "geojson",
            "resultOffset": result_offset,
            "resultRecordCount": page_size,
            "geometryPrecision": 6,
        }

        resp = requests.get(NFHL_LAYER_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "features" not in data:
            raise RuntimeError(f"Unexpected response: {json.dumps(data)[:400]} ...")

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        result_offset += len(features)

        time.sleep(sleep)


        if len(features) < page_size:
            break

    return {
        "type": "FeatureCollection",
        "features": all_features,
    }

def _write_geojson(path: Path, feature_collection: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, ensure_ascii=False)

def _merge_feature_collections(paths):
    merged = {"type": "FeatureCollection", "features": []}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            gj = json.load(f)
        merged["features"].extend(gj.get("features", []))
    return merged

def fetch_nfhl_bbox(min_lat, min_lon, max_lat, max_lon, fields=tuple(DEFAULT_FIELDS),
                    grid_rows=1, grid_cols=1, page_size=2000, out_dir="data/raw/nfhl",
                    prefix="bbox", dry_run=False):
    stamp = datetime.utcnow().strftime("%Y%m%d")
    bbox_dirname = f"{prefix}_{min_lat}_{min_lon}_{max_lat}_{max_lon}"
    base_dir = Path(out_dir) / stamp / _safe_filename(bbox_dirname)
    _ensure_dir(base_dir)

    tile_files = []

    for r, c, (tmin_lat, tmin_lon, tmax_lat, tmax_lon) in _tile_bounds(
        min_lat, min_lon, max_lat, max_lon, grid_rows, grid_cols
    ):
        tile_name = f"tile_r{r}_c{c}.geojson"
        tile_path = base_dir / tile_name

        if dry_run:
            print(f"[DRY RUN] Would fetch: r={r} c={c} bbox=({tmin_lat},{tmin_lon},{tmax_lat},{tmax_lon}) -> {tile_path}")
            continue

        print(f"Fetching r={r} c={c} …")
        fc = _fema_query_bbox(
            tmin_lat, tmin_lon, tmax_lat, tmax_lon, fields=fields, page_size=page_size
        )
        _write_geojson(tile_path, fc)
        print(f"  saved {len(fc.get('features', []))} features → {tile_path}")
        tile_files.append(tile_path)

    if not dry_run and tile_files:
        merged = _merge_feature_collections(tile_files)
        merged_path = base_dir / "merged.geojson"
        _write_geojson(merged_path, merged)
        print(f"Merged {sum(1 for _ in merged['features'])} features → {merged_path}")

    return str(base_dir)

def main():
    ap = argparse.ArgumentParser(description="Fetch FEMA NFHL flood zones as GeoJSON.")
    ap.add_argument("--min-lat", type=float, required=True)
    ap.add_argument("--min-lon", type=float, required=True)
    ap.add_argument("--max-lat", type=float, required=True)
    ap.add_argument("--max-lon", type=float, required=True)
    ap.add_argument("--fields", type=str, default=",".join(DEFAULT_FIELDS),
                    help="Comma-separated attribute list, or * for all fields.")
    ap.add_argument("--grid-rows", type=int, default=1, help="Split bbox into N rows (tiling).")
    ap.add_argument("--grid-cols", type=int, default=1, help="Split bbox into N cols (tiling).")
    ap.add_argument("--page-size", type=int, default=2000, help="Records per page for pagination.")
    ap.add_argument("--out-dir", type=str, default="data/raw/nfhl")
    ap.add_argument("--prefix", type=str, default="bbox")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",")] if args.fields != "*" else "*"

    fetch_nfhl_bbox(
        min_lat=args.min_lat,
        min_lon=args.min_lon,
        max_lat=args.max_lat,
        max_lon=args.max_lon,
        fields=fields,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        page_size=args.page_size,
        out_dir=args.out_dir,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()