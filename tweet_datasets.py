import os
import csv
import random
from itertools import islice

import numpy as np
import torch
from torch.utils.data import Dataset

class TweetData(Dataset):
    """
    Put language datasets into PyTorch format
    """

    def __init__(self, emb_dir, batchsz, n_way, k_shot, k_query, mode='', lang_only=False, disk_load=False):
        """
        emb_dir: Directory with the embeddings
        mode: train, dev, or test
        batchsz: batch size of sets
        n_way
        k_shot
        k_query

        if mode is 'train' then only take 'train' and 'dev' data
        if mode is 'test' then oly take 'test' data
        if lang_only then only use language as label, not sentiment
        if disk_load then load samples from disk (used for large dim embeddings)
        """

        # Data location
        self.emb_dir = emb_dir
        self.lang_only = lang_only
        self.disk_load = disk_load
        self.mode = mode

        # User provided params
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        # Used for batch selection
        self.setsz = self.n_way * self.k_shot
        self.querysz = self.n_way * self.k_query

        # Loading the data
        self.data = []

        # { lbl name : idx in self.data }
        self.class_idx = {}
        # { idx : lbl name }
        self.idx_to_lbl = {}
        csvdata = self.load_data()
        for i, (lbl, twts) in enumerate(csvdata.items()):
            self.data.append(twts)
            self.class_idx[lbl] = i
            self.idx_to_lbl[i] = lbl
        self.cls_num = len(self.class_idx)
        
        self.create_batch()

    def load_data(self):
        dict_labels = {}
        for fn in os.listdir(self.emb_dir):
            if fn == '.placeholder':
                continue
            fdata, ext = fn.split('.')
            ftype, lang, dset = fdata.split('_')
           
            if self.mode == 'train' and dset == 'test':
                continue
            elif self.mode == 'test' and dset in ['train', 'dev']:
                continue
 
            with open(os.path.join(self.emb_dir, fn), 'r') as edfo:
                data_iter = None
                if self.disk_load:
                    data_iter = edfo
                else:
                    data_iter = csv.reader(edfo, delimiter=',')
                row_num = 0
                while True:
                    twt_emb = []
                    class_lbl = None

                    row = None
                    cur_filepos = None
                    if self.disk_load:
                        cur_filepos = data_iter.tell()
                        row = data_iter.readline()
                        if len(row) == 0:
                            break
                        reader = csv.reader([row], delimiter=',')
                        row = next(reader)
                    else:
                        try:
                            row = next(data_iter)
                        except StopIteration:
                            break
                    
                    twt_data = None
                    sent_lbl = None
                    if len(row) == 2:
                        # fasttext embs
                        if not self.disk_load:
                            twt_data = [ float(v) for v in row[0][1:-1].split(' ') if len(v) > 0 ]
                            twt_data = np.array(twt_data)
                            sent_lbl = int(row[1])
                        else:
                            twt_data = ( fn, cur_filepos )
                            sent_lbl = int(row[1])
                            #sent_lbl = 0 if row[-1] == '0' else int(row[-2:])
                    else:
                        # laser embs
                        if not self.disk_load:
                            twt_data = np.array([ float(v) for v in row[:-1] ])
                            sent_lbl = int(float(row[-1]))
                        else:
                            twt_data = ( fn, cur_filepos )
                            sent_lbl = int(float(row[-1]))
                            #print('read row', row, flush=True)
                            #sent_lbl = 0 if row[-1] == '0' else int(row[-2:])

                    if self.lang_only:
                        class_lbl = lang
                    elif sent_lbl == -1:
                        class_lbl = lang + '_neg'
                    elif sent_lbl == 0:
                        class_lbl = lang + '_neu'
                    elif sent_lbl == 1:
                        class_lbl = lang + '_pos'
                    else:
                        raise ValueError

                    if class_lbl in dict_labels:
                        dict_labels[class_lbl].append(twt_data)
                    else:
                        dict_labels[class_lbl] = [ twt_data ]

                    row_num += 1
        return dict_labels

    def create_batch(self):

        # dim: [ self.batchsz, self.n_way, self.k_shot ]
        self.support_x_batch = [] # support set batch
        self.support_y_batch = [] # the labels
        # dim: [ self.batchsz, self.n_way, self.k_query ]
        self.query_x_batch = [] # query set batch
        self.query_y_batch = []
        for b in range(self.batchsz):
            # Select n_way classes randomly
            selected_cls_idx = np.random.choice(self.cls_num, self.n_way, False)
            np.random.shuffle(selected_cls_idx)
            support_x = []
            support_y = []
            query_x = []
            query_y = []
            for cls_idx in selected_cls_idx:
                # Select k_shot + k_query samples from each class
                cls_data_size = -1
                cls_data_size = len(self.data[cls_idx])

                selected_twts_idx = np.random.choice(
                    cls_data_size, self.k_shot + self.k_query, False)
                np.random.shuffle(selected_twts_idx)
                index_train = np.array(selected_twts_idx[:self.k_shot]) # data indices for train
                index_test = np.array(selected_twts_idx[self.k_shot:]) # for test

                # if disk load then only save the indices
                spt_x_data = np.array(self.data[cls_idx])[index_train].tolist()
                qry_x_data = np.array(self.data[cls_idx])[index_test].tolist()
                support_x.append(spt_x_data)
                support_y.append(
                    np.repeat(cls_idx, self.k_shot).tolist())
                query_x.append(qry_x_data)
                #if len(query_x[-1]) != self.k_query:
                #    print('query', self.k_query, 'shot', self.k_shot, flush=True)
                #    print('selected len', len(selected_twts_idx), flush=True)
                #    print('index train len', len(index_train), flush=True)
                #    print('index test len', len(index_test), flush=True)
                #    print('query len', len(query_x[-1]), flush=True)
                #    print('total data len', len(self.data[cls_idx]), flush=True)
                query_y.append(
                    np.repeat(cls_idx, self.k_query).tolist())

            # shuffle support and query sets
            random.shuffle(support_x)
            random.shuffle(support_y)
            random.shuffle(query_x)
            random.shuffle(query_y)

            # train data
            self.support_x_batch.append(support_x) # [ [batch1], [batch2],... ]
            self.support_y_batch.append(support_y) # [ [lbl1,lbl1,...], ... ]
            # test data
            self.query_x_batch.append(query_x)
            self.query_y_batch.append(query_y)
        
    def read_from_disk(self, flat_x, fn_fo_map):
        # Will change flat_x in place
        for i, (fn, file_pos) in enumerate(flat_x):
            if fn not in fn_fo_map:
                fo = open(os.path.join(self.emb_dir, fn), 'r')
                fn_fo_map[fn] = fo 
            fo = fn_fo_map[fn]
            fo.seek(int(file_pos))
            line = fo.readline()
            row_reader = csv.reader([ line ], delimiter=',')
            row = next(row_reader)
 
            twt_emb = None
            if len(row) == 2:

                # non-laser embs
                twt_emb = [ float(v) for v in row[0][1:-1].split(' ') if len(v) > 0 ]
                twt_emb = np.array(twt_emb)
            elif len(row) == 1025:

                # laser embs
                twt_emb = [ float(v) for v in row[:-1] ]
            else:
                print('file', fn, 'pos', file_pos)
                print(len(row), flush=True)
                print(row, flush=True)
                raise ValueError

            if len(twt_emb) != 1024:
                print('invalid length', flush=True)
                print('file', fn, 'pos', file_pos)
                print(len(twt_emb), flush=True)
                print(row, flush=True)

            flat_x[i] = twt_emb

        return flat_x, fn_fo_map

    def __getitem__(self, idx):
        """
        Get the idx indexed batch from the full data
        0 <= idx <= batchsz - 1
        """

        # self.support_x_batch[idx] has dim [ self.n_way, self.k_shot ]
        # flattened to be [ self.n_way * self.k_shot ]
        flatten_support_x = [ twt_data
            for sublist in self.support_x_batch[idx] for twt_data in sublist ]

        # flattened to be [ self.n_way * self.k_query ]
        #print('query total shape', np.array(self.query_x_batch[idx]).shape, flush=True)
        flatten_query_x = [ twt_data
            for sublist in self.query_x_batch[idx] for twt_data in sublist ]

        # if disk load need to convert twt_data into actual embedding
        if self.disk_load:
            fn_fo_map = { } # { fn : fo }

            flatten_support_x, fn_fo_map = self.read_from_disk(flatten_support_x, fn_fo_map)
            flatten_query_x, fn_fo_map = self.read_from_disk(flatten_query_x, fn_fo_map)

            for fo in fn_fo_map.values():
                fo.close()

        support_y = np.array([ lbl_idx
            for sublist in self.support_y_batch[idx] for lbl_idx in sublist ]).astype(np.int32)
        query_y = np.array([ lbl_idx
            for sublist in self.query_y_batch[idx] for lbl_idx in sublist ]).astype(np.int32)

        #print('flatten query x shape', np.array(flatten_query_x).shape, flush=True)
        #print('query y shape', np.array(query_y).shape, flush=True)

        # put samples into tensors
        emb_size = len(flatten_support_x[0])
        # [ n_way * k_shot (setsz) , 1, emb_size ]
        support_x = torch.FloatTensor(
            np.array(flatten_support_x).reshape((self.setsz, 1, emb_size))
        )
        # [ n_way * k_query (querysz) , 1, emb_size ]
        #print('query shape', np.array(flatten_query_x).shape, flush=True)
        #print('query x elem type', type(flatten_query_x[0]), flush=True)
        #print('query x elem [0] type', type(flatten_query_x[0][0]), flush=True)
        for vec in flatten_query_x:
            if len(vec) != 1024:
                print(flatten_query_x, flush=True)
            assert len(vec) == 1024
        #print('queryx elem size', len(flatten_query_x[0]), flush=True)
        query_x = torch.FloatTensor(
            np.array(flatten_query_x).reshape((self.querysz, 1, emb_size))
        )

        # get relative labels for the batch
        # sampled classes same in support and query
        # label ranges from 0 to n-way
        unique_lbls = np.unique(support_y)
        random.shuffle(unique_lbls)

        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique_lbls):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        return self.batchsz

if __name__ == '__main__':
    tweets = TweetData('embs', n_way=5, k_shot=1, k_query=1, batchsz=1000)
    for i, set_ in enumerate(tweets):
        support_x, support_y, query_x, query_y = set_

        print('batch', i)
        print('support data', support_x[:5])
        print('support labels', support_y[:5])
        print('query data', query_x[:5])
        print('query labels', query_y[:5])

        if i == 10:
            break

