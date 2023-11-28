import os
import csv

import fasttext
import numpy as np

"""
Take the cleaned datasets
Convert them into embedding files
"""

if __name__ == '__main__':

    # If this is true then use already trained models if they exist
    # Otherwise retrain
    cache_models = True

    data_dir = 'data'
    emb_models_dir = 'emb_models'
    embs_dir = 'embs'

    languages = { }

    # Get the set of languages
    for lang_group in os.listdir(data_dir):
        lang_dir = os.path.join(data_dir, lang_group)

        languages[lang_group] = set()
        for lf in os.listdir(lang_dir):
            lf_path = os.path.join(lang_dir, lf)
 
            fn, ext = lf.split('.')

            if lf == '.placeholder' or ext == 'tmp':
                continue

            fsplit = lf.split('_')
            if len(fsplit) == 2:
                continue
            ftype, lang, dset = lf.split('_')

            languages[lang_group].add(lang)

    # Get the models
    for ldir, lset in languages.items():
        lang_dir = os.path.join(data_dir, ldir)
        for lang in lset:
            model_fp = os.path.join(emb_models_dir, lang + '.bin')
            if cache_models and os.path.exists(model_fp):
                continue
            train_fp = os.path.join(lang_dir, 'clean_{l}_train.csv'.format(l=lang))

            # Create temp file

            temp_fp = os.path.join(lang_dir, '{l}_train_ft.tmp'.format(l=lang))
            with open(temp_fp, 'w+') as tf:
                train_sentences = []
                train_fo = open(train_fp, 'r')
                tr_data = csv.reader(train_fo, delimiter=',')
                sentences = []
                for row in tr_data:
                    sentences.append(row[0])
                train_fo.close()
                tf.write('\n'.join(sentences))
                train_fo.close()

            print('Training FastText embedding model for', lang)
            model = fasttext.train_unsupervised(temp_fp, model='skipgram')

            # Delete temp file
            os.remove(temp_fp)
            
            print('Saving model')
            model.save_model(model_fp)

    # Get the csv embeddings
    for ldir, lset in languages.items():
        lang_dir = os.path.join(data_dir, ldir)
        for lang in lset:
            dsets = ['train', 'test', 'dev']
            train_fp = os.path.join(lang_dir, 'clean_{l}_train.csv'.format(l=lang))
            
            # Get the language model
            print('Loaded', lang, 'FastText model')
            model_fp = os.path.join(emb_models_dir, lang + '.bin')

            ft_model = fasttext.load_model(model_fp)
            
            # Get the embeddings
            for dset in dsets:
                print('Getting embeddings for lang', lang, 'set', dset)
                dset_fp = os.path.join(lang_dir,
                    'clean_{l}_{d}.csv'.format(l=lang,d=dset))
                emb_fp = os.path.join(embs_dir,
                    'emb_{l}_{d}.csv'.format(l=lang, d=dset))
                emb_fo = open(emb_fp, 'w+')
                emb_rows = []
                with open(dset_fp, 'r') as dfo:
                    dfo_reader = csv.reader(dfo, delimiter=',')
                    emb_writer = csv.writer(emb_fo, delimiter=',')

                    for row in dfo_reader:
                        sent_emb = ft_model.get_sentence_vector(row[0])
                        emb_str = np.array2string(sent_emb)
                        emb_writer.writerow([ emb_str, row[1] ])
                emb_fo.close()
                print('Embeddings saved')

