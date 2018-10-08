#!/bin/bash

#$ -N main.py
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1

# Array job: -t <range>
#$ -t 1

#$ -l h=node06

#$ -v DISPLAY

#$ -o /data/cvg/lukas/learning-object-representations-by-mixing-scenes/main.log

#$ -m ea
#$ -M lukas.zbinden@unifr.ch

python main.py 2>&1 | tee -a main.log &
