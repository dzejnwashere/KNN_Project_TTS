#!/bin/bash
# this script copies all mp3 and wav files recursively to target dir
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <search_path> <destination_path>"
  exit 1
fi
SEARCH_PATH="$1"
DEST_PATH="$2"
# if destination does not exist, creates it
mkdir -p "$DEST_PATH"
find "$SEARCH_PATH" -type f \( -iname "*.mp3" -o -iname "*.wav" \) -exec bash -c '
  for file; do
    echo "Copying: $file to: '"$DEST_PATH"'"
    cp "$file" '"$DEST_PATH"'
  done
' bash {} +