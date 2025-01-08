#!/bin/bash

# Set error handling
set -e

# Get the directory where the script is located
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source ./venv/bin/activate

# Run batch processing script
python batch_processing.py >> batch_processing.log 2>&1

# Add timestamp to log
echo "Batch processing completed at $(date)" >> batch_processing.log

# Send email notification
echo "Batch processing completed on $(date)" | /usr/bin/mail -s "Batch Processing Report" david-patrick-philippe.lupau@iu-study.org
