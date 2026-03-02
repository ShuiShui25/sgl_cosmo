#!/usr/bin/env bash
# run_all.sh — run five Python scripts in order

# Exit immediately if any command fails
set -e

# (Optional) Print each command before running
set -x

# Run each Python script

python dis_ratio_check_0_0.02.py
python dis_ratio_check_0_0.04.py
python dis_ratio_check_0_0.08.py
#python dis_ratio_check+0.1.py
#python dis_ratio_check-0.01.py
#python dis_ratio_check-0.02.py
#python dis_ratio_check-0.04.py
#python dis_ratio_check-0.1.py

echo "All done!"

