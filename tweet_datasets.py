import os
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

class TweetData(Dataset):
    """
    Put language datasets into PyTorch format
    """

    def __init__(self, emb_dir, batchsz, n_way, k_shot, k_query):
        """
        emb_dir: Directory with the embeddings
        mode: train, dev, or test
        batchsz: batch size of sets
        n_way
        k_shot
        k_query
        """

        # Data location
        self.emb_dir = emb_dir

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
        csvdata = self.load_data()
        for i, (lbl, twts) in enumerate(csvdata.items()):
            self.data.append(twts)
            self.class_idx[lbl] = i
        self.cls_num = len(self.class_idx)
        
        self.create_batch()

    def load_data(self):
        dict_labels = {}
        for fn in os.listdir(self.emb_dir):
            if fn == '.placeholder':
                continue
            fdata, ext = fn.split('.')
            ftype, lang, dset = fdata.split('_')
            
            with open(os.path.join(self.emb_dir, fn), 'r') as edfo:
                csvreader = csv.reader(edfo, delimiter=',')
                for row in csvreader:
                    twt_emb = [ float(v) for v in row[0][1:-1].split(' ') if len(v) > 0 ]
                    twt_emb = np.array(twt_emb)
                    assert len(twt_emb) == 100
                    sent_lbl = int(row[1])
                    class_lbl = None
                    if sent_lbl == -1:
                        class_lbl = lang + '_neg'
                    elif sent_lbl == 0:
                        class_lbl = lang + '_neu'
                    elif sent_lbl == 1:
                        class_lbl = lang + '_pos'
                    else:
                        raise ValueError

                    if class_lbl in dict_labels:
                        dict_labels[class_lbl].append(twt_emb)
                    else:
                        dict_labels[class_lbl] = [ twt_emb ]
        return dict_labels

    def create_batch(self):
        self.support_x_batch = [] # support set batch
        self.support_y_batch = [] # the labels
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
                selected_twts_idx = np.random.choice(
                    len(self.data[cls_idx]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_twts_idx)
                index_train = np.array(selected_twts_idx[:self.k_shot]) # data indices for test
                index_test = np.array(selected_cls_idx[self.k_shot:]) # for train
                support_x.append(
                    np.array(self.data[cls_idx])[index_train].tolist())
                support_y.append(
                    np.repeat(cls_idx, self.k_shot).tolist())
                query_x.append(
                    np.array(self.data[cls_idx])[index_test].tolist())
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
        
    def __getitem__(self, idx):
        """
        Get the idx indexed batch from the full data
        0 <= idx <= batchsz - 1
        """

        # flatten out samples and labels
        flatten_support_x = [ twt_emb
            for sublist in self.support_x_batch[idx] for twt_emb in sublist ]
        support_y = np.array([ lbl_idx
            for sublist in self.support_y_batch[idx] for lbl_idx in sublist ]).astype(np.int32)

        flatten_query_x = [ twt_emb
            for sublist in self.query_x_batch[idx] for twt_emb in sublist ]
        query_y = np.array([ lbl_idx
            for sublist in self.query_y_batch[idx] for lbl_idx in sublist ]).astype(np.int32)

        # put samples into tensors
        emb_size = len(flatten_support_x[0])
        support_x = torch.FloatTensor(self.setsz, emb_size)
        query_x = torch.FloatTensor(self.setsz, emb_size)

        # get relative labels for the batch
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

