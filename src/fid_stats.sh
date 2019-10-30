#!/bin/bash


# grep "\"fid\"" logs/$1/metrics/results/log*json | awk '{print $3}' | awk -F, '{print $1}' | sort -rn

echo -e "\nNumber of epochs:"
grep "\"fid\"" logs/$1/metrics/results/log*json | wc -l

echo -e "\nStats of last 10 FID values:"
grep "\"fid\"" logs/$1/metrics/results/log*json | awk '{print $3}' | awk -F, '{print $1}' | tail -10 | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "Min: %f\tMax: %f\tAverage: %f\n", min, max, sum/NR}'

echo -e "\nStats of all FID values:"
grep "\"fid\"" logs/$1/metrics/results/log*json | awk '{print $3}' | awk -F, '{print $1}' | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "Min: %f\tMax: %f\tAverage: %f\n", min, max, sum/NR}'

echo -e ""
