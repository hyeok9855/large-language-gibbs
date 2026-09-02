#!/bin/bash
# Download the assets needed to *score* the DAT (no LLM involved):
#   - glove.840B.300d.txt  GloVe embeddings (~2GB zipped, ~5.3GB unzipped)
#   - words.txt            dictionary from Olson et al.'s reference scorer
#
# Usage: bash divergent_association_task/download_assets.sh
#
# Files appear under their final names only when complete, so the existence
# guards stay safe across interrupted runs; the partial zip is kept for resume.
set -euo pipefail

ASSETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/assets"
mkdir -p "$ASSETS_DIR"
cd "$ASSETS_DIR"

if [ ! -f words.txt ]; then
    echo "Downloading words.txt (Olson et al. dictionary)..."
    curl -fL --retry 3 --retry-delay 5 -o words.txt.tmp \
        https://raw.githubusercontent.com/jayolson/divergent-association-task/main/words.txt
    mv words.txt.tmp words.txt
fi

if [ ! -f glove.840B.300d.txt ]; then
    echo "Downloading glove.840B.300d.zip (~2GB, resumable)..."
    curl -fL -C - --retry 3 --retry-delay 5 -o glove.840B.300d.zip \
        https://nlp.stanford.edu/data/glove.840B.300d.zip
    rm -rf extract.tmp
    unzip -o -q glove.840B.300d.zip -d extract.tmp
    mv extract.tmp/glove.840B.300d.txt glove.840B.300d.txt
    rm -rf extract.tmp glove.840B.300d.zip
fi
