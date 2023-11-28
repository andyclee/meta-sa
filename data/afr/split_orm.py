import csv
from math import floor

dev = 'clean_afr-tir_dev.csv'
test = 'clean_afr-tir_test.csv'

total = []

with open(dev, 'r') as do:
    rdr = csv.reader(do, delimiter=',')
    for row in rdr:
        total.append(row)

with open(test, 'r') as to:
    rdr = csv.reader(to, delimiter=',')
    for row in rdr:
        total.append(row)

train_s = 0.6
dev_s = 0.2
test_s = 0.2

examples = len(total)

train_cnt = floor(train_s * examples)
test_cnt = floor(test_s * examples)
dev_cnt = floor(dev_s * examples)

train_cnt += examples - (train_cnt + test_cnt + dev_cnt)

print(train_cnt, test_cnt, dev_cnt)

with open('clean_afr-tir_train.csv', 'w+') as fo:
    wrtr = csv.writer(fo, delimiter=',')
    for row in total[:train_cnt]:
        wrtr.writerow(row)

with open('clean_afr-tir_dev.csv', 'w+') as fo:
    wrtr = csv.writer(fo, delimiter=',')
    for row in total[train_cnt:train_cnt + dev_cnt]:
        wrtr.writerow(row)

with open('clean_afr-tir_test.csv', 'w+') as fo:
    wrtr = csv.writer(fo, delimiter=',')
    for row in total[train_cnt + dev_cnt:]:
        wrtr.writerow(row)

