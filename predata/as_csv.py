import os
import csv

data_dir = 'AfriSenti'

for lang in os.listdir(data_dir):
    lang_dir = os.path.join(data_dir, lang)
    if not os.path.isdir(lang_dir):
        continue
    print('procesing lang', lang)
    for dataset in os.listdir(lang_dir):
        set_type, ext = dataset.split('.')
        out_fn = 'afr-{l}_{stype}.csv'.format(l=lang, stype=set_type)
        outf = open(out_fn, 'w+')
        with open(os.path.join(data_dir, lang, dataset), 'r') as df:
            set_reader = csv.reader(df, delimiter='\t')
            next(set_reader, None)
            out_writer = csv.writer(outf, delimiter=',')
            for row in set_reader:
                if lang == 'orm':
                    out_writer.writerow([row[1], row[2]])
                else:
                    out_writer.writerow([row[0], row[1]])
        outf.close()
