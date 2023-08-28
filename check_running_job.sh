#!/bin/bash

first_arg=$1

while true; do
    # Add your desired command here
    # For example, let's echo the current time
current_time=$(date "+%d-%m-%Y %H:%M:%S")

echo "Current time: $current_time"
squeue -u emok
echo

# Sleep for 2 minutes (120 seconds) before running the command again
    sleep 120
done
