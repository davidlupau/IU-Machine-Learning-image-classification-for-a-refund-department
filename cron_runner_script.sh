#!/bin/bash

# Activate virtual environment
source ./venv/bin/activate

# Run batch processing script
python batch_processing.py

# Send email notification
echo "Batch processing completed on $(date)" | /usr/bin/mail -s "Batch Processing Report" david-patrick-philippe.lupau@iu-study.org