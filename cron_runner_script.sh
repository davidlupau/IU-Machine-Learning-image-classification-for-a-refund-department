#!/bin/bash
set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

echo "Activating virtual environment..." >> "$PROJECT_DIR/batch_processing.log"
source ./venv/bin/activate

echo "Running batch processing..." >> "$PROJECT_DIR/batch_processing.log"
python3 batch_processing.py >> "$PROJECT_DIR/batch_processing.log" 2>&1

echo "Batch processing completed at $(date)" >> "$PROJECT_DIR/batch_processing.log"