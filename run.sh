#!/bin/bash

URL="https://techassessment.blob.core.windows.net/aiip6-assessment-data/bmarket.db"
DEST_DIR="/data"
sudo mkdir -p "$DEST_DIR"
sudo curl -o "$DEST_DIR/bmarket.db" "$URL"
python src/main.py