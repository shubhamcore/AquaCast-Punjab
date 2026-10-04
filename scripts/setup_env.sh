#!/usr/bin/env bash
# Reinstall the Python environment.
#
# The sandbox /tmp is a ~993 MB tmpfs, which is smaller than the torch wheel,
# so pip's default temp dir overflows ("No space left on device") even when the
# root volume has 20 GB free. Point TMPDIR at the root volume and pull torch
# from the CPU index (~5x smaller than the default CUDA build).
set -euo pipefail

BIGTMP="${BIGTMP:-/tmp/aquacast-bigtmp}"
mkdir -p "$BIGTMP"
export TMPDIR="$BIGTMP"

echo "== torch (CPU index) =="
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

echo "== geospatial + dashboard + deck =="
pip install --no-cache-dir \
    geopandas shapely pyproj pyogrio rasterio \
    folium streamlit-folium \
    scikit-learn streamlit plotly \
    openpyxl xlsxwriter gdown \
    python-pptx matplotlib

echo "== verify =="
python3 - <<'PY'
import importlib
missing = []
for m in ("numpy","pandas","scipy","torch","geopandas","shapely","pyproj","pyogrio",
          "rasterio","folium","streamlit","plotly","sklearn","matplotlib","pptx"):
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print("missing:", missing or "none")
PY
