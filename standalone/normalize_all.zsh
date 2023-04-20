#!/bin/zsh

# list of files to process
files=(10p4.pkl  11p4.pkl  3p6.pkl  4p4.pkl  4p6.pkl  5p2.pkl  6p4.pkl  6p6.pkl  7p2.pkl  7p8.pkl  8p4.pkl)

# loop through each file
for file in "${files[@]}"
do
  # execute one-line executable with filename
  screen -S "$file" srun -c 2 --mem=32gb -G 1 -p gpu-preempt --pty /usr/bin/python3 /home/kkreth_umassd_edu/cgan/standalone/15_normalization.py /work/pi_bseyedaghazadeh_umassd_edu/raw_input/"$file"
done