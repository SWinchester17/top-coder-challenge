#!/bin/bash

# Black Box Challenge - Reimbursement Calculator
# Based on Machine Learning (Gradient Boosting)

DAYS=$1
MILES=$2
RECEIPTS=$3

# Validate inputs
if [ -z "$DAYS" ] || [ -z "$MILES" ] || [ -z "$RECEIPTS" ]; then
    echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

# Call Python script
python3 ml_calculate.py "$DAYS" "$MILES" "$RECEIPTS" 