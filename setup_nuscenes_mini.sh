#!/usr/bin/env bash
set -e
TARGET_DIR="/scratch/m25csa031/nuscenes"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "=== Starting nuScenes v1.0-mini Download ==="
wget -c --progress=dot:giga "https://www.nuscenes.org/data/v1.0-mini.tgz" -O v1.0-mini.tgz

echo "=== Extracting archive ==="
tar -xf v1.0-mini.tgz
rm -f v1.0-mini.tgz

echo "=== Creating symlinks in project data/nuscenes ==="
PROJ_DATA="/csehome/m25csa031/nuscenes_vlm_phase1/data/nuscenes"
mkdir -p "$PROJ_DATA"
ln -sfn "$TARGET_DIR/v1.0-mini" "$PROJ_DATA/v1.0-mini"
ln -sfn "$TARGET_DIR/samples"   "$PROJ_DATA/samples"
ln -sfn "$TARGET_DIR/sweeps"    "$PROJ_DATA/sweeps"
[ -d "$TARGET_DIR/maps" ] && ln -sfn "$TARGET_DIR/maps" "$PROJ_DATA/maps"

echo "=== nuScenes v1.0-mini Setup Complete ==="
