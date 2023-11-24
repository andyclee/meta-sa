import os
import csv
from math import floor

eng_file = 'Sentiment140/eng_tweets.csv'

train = 0.6
dev = 0.2
test = 0.2

assert train + dev + test == 1.0, 'splits must sum to 1'

csv_data = []
with open(eng_file, 'r', encoding='ISO-8859-1') as ef:
    ef_reader = csv.reader(ef, delimiter=',')
    csv_data = list(ef_reader)

data_size = len(csv_data)
train_size = floor(data_size * train)
dev_size = floor(data_size * dev)
test_size = floor(data_size * test)

leftover = data_size - train_size - dev_size - test_size
train_size += leftover

train_data = csv_data[:train_size]
dev_data = csv_data[train_size : train_size + dev_size]
test_data = csv_data[train_size + dev_size : ]

with open('en_train.csv', 'w+') as et:
    dt_writer = csv.writer(et, delimiter=',')
    for row in train_data:
        dt_writer.writerow([ row[5], row[0] ])

with open('en_dev.csv', 'w+') as et:
    dt_writer = csv.writer(et, delimiter=',')
    for row in dev_data:
        dt_writer.writerow([ row[5], row[0] ])

with open('en_test.csv', 'w+') as et:
    dt_writer = csv.writer(et, delimiter=',')
    for row in test_data:
        if len(row) < 2:
            continue
        dt_writer.writerow([ row[5], row[0] ])

