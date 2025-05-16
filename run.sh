#!/bin/bash

URL="https://techassessment.blob.core.windows.net/aiip6-assessment-data/bmarket.db"
DEST_DIR="/data"
mkdir -p "$DEST_DIR"
curl -o "$DEST_DIR/bmarket.db" "$URL"
python src/main.py