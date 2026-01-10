#!/bin/bash
#
# AetherMind Daily Training Runner
#
# Runs a specific benchmark family for a set number of "days" to simulate
# progressive learning over time. Each day's run is saved to a separate file.
#
# Usage:
#   ./run_daily_training.sh [family] [days]
#
# Example:
#   # Run the 'gsm' family for 5 days
#   ./run_daily_training.sh gsm 5
#

# --- Configuration ---
FAMILY=${1:-gsm}    # Default to 'gsm' if no family is provided
TOTAL_DAYS=${2:-5}  # Default to 5 days if not specified
BASE_RESULTS_DIR="benchmarks/results"

# Any additional arguments to pass to the progressive_runner
# e.g., --yolo, --chunk-size 50
EXTRA_ARGS="${@:3}"

# --- Main Loop ---
echo "🚀 Starting Daily Training Cycle for family: '$FAMILY'"
echo "   Total days to simulate: $TOTAL_DAYS"
echo "   Extra arguments: $EXTRA_ARGS"
echo "--------------------------------------------------"

for (( i=1; i<=$TOTAL_DAYS; i++ ))
do
    # Format day with leading zero
    DAY_NUM=$(printf "%02d" $i)
    
    # Define the output path for the current day
    OUTPUT_FILE="${BASE_RESULTS_DIR}/day_${DAY_NUM}_${FAMILY}_progressive.json"
    
    echo ""
    echo "--- Day $DAY_NUM / $TOTAL_DAYS ---"
    echo "   Family: $FAMILY"
    echo "   Saving results to: $OUTPUT_FILE"
    
    # Construct the command
    COMMAND="python -m benchmarks.progressive_runner --family $FAMILY --output $OUTPUT_FILE --auto $EXTRA_ARGS"
    
    echo "   Executing: $COMMAND"
    echo ""
    
    # Execute the command
    eval $COMMAND
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "❌ Error on Day $DAY_NUM. Aborting training cycle."
        exit 1
    fi
    
    echo "✅ Day $DAY_NUM complete."
    echo "--------------------------------------------------"
done

echo "🎉 Daily Training Cycle finished for family: '$FAMILY'"
echo "   All $TOTAL_DAYS days completed successfully."

