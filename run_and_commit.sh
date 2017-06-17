#!/bin/bash

# Script to run main.py and automatically commit results when done with a supplied commit message

echo "Running main and committing & pushing results..."
python3 main.py
git add results/*
if [[ $# -ne 1 ]]; then 
    git commit -m "Automated commit of run results"
else
    git commit -m "Automated commit of run results - $1"
fi
git push origin $(git symbolic-ref --short -q HEAD)
echo "...automated commit & push finished!"
