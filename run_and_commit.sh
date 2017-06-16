#!/bin/bash

echo "Running main and committing & pushing results..."
#python3 main.py
git add results/*
git commit -m "Automated commit of results"
git push origin $(git symbolic-ref --short -q HEAD)
echo "...automated commit & push finished!"
