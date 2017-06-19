#!/bin/bash

echo "Generating short dict"
cat train_pos.txt train_neg.txt cleared_test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt
echo "Generating full dict"
cat train_pos_full.txt train_neg_full.txt cleared_test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
echo "Generating test dict"
cat cleared_test_data.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > test_vocab.txt
