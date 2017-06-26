#!/bin/bash

# Script to run main.py and automatically commit results when done with a supplied commit message
echo "[Data preparation] Generating vocabulary and clearing test set..."
cd twitter-datasets
./build.sh
cd ..

echo "[Main run] Running main.py and committing & pushing results..."
#python3 main.py
echo "[Main run] ...finished running main.py, committing & pushing results..."
#git pull origin $(git symbolic-ref --short -q HEAD)
#git add results/*
#if [[ $# -ne 1 ]]; then 
#    git commit -m "Automated commit of run results"
#else
#    git commit -m "Automated commit of run results - $1"
#fi
#git push origin $(git symbolic-ref --short -q HEAD)
echo "[Main run] ...automated commit & push finished!"
