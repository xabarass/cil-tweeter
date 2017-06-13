#!/bin/bash

echo "Generating short dict"
cat train_pos.txt train_neg.txt test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
echo "Generating full dict"
cat train_pos_full.txt train_neg_full.txt test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
