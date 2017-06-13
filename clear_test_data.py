def remove_index(line):
    comma_pos = line.find(',')
    tweet_id = int(line[:comma_pos])
    tweet = line[comma_pos + 1:]
    return tweet


with open('test_data.txt','r') as test_data, open('cleared_test_data.txt','w') as output:
    for line in test_data:
        cleared_line=remove_index(line)
        output.write(cleared_line)


