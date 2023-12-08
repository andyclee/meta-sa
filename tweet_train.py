import os
import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from tweet_datasets import TweetData
from meta import Meta

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # setup the layers here
    config = None

    config_cnn = [
        # (layer_name, [params]),
        ('conv1d', [32, 1, 3, 1, 'same']),
        ('elu', [False]),
        ('conv1d', [32, 32, 3, 1, 'same']),
        ('elu', [False]),
        #('conv1d', [32, 32, 3, 1, 'same']),
        #('relu', [False]),
        ('max_pool1d', [2]),
        #('globalmax_pool1d', []),
        ('dense', [128, int(32 * args.embdim / 2)]),
        ('tanh', []),
        #('relu', [False]),
        ('dropout', [0.5, False]),
        ('dense', [args.n_way, 128])
#        ('softmax', []) # softmax applied in forward/finetunning
    ]

    # LSTMS all have dropout probability of 0.05
    config_lstm = [
                    # [ in_size, hidden_size, dropout ] 
        ('bi-lstm', [args.embdim, 50, 0.05]), 
        ('bi-lstm', [50 * 2, 20, 0.05]),
                    # [ out_size, in_size ]
        ('dense', [20, 20 * 2]),
        ('elu', [False]),
        ('dropout', [0.5, False]),
        ('dense', [args.n_way, 20])
    ]

    if args.arch == 'cnn':
        config = config_cnn
    elif args.arch == 'lstm':
        config = config_lstm
    else:
        raise NotImplementedError

    device = torch.device('cuda')
    #device = torch.device('cpu')

    # Get Meta learner from meta
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x : x.requires_grad, maml.parameters())
    num = sum(map(lambda x : np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    emb_dir = args.data_dir

    # batchsz is total episode number
    tweets_train = TweetData(emb_dir,
        n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
        batchsz=args.train_batchsz, disk_load=args.disk_load, mode='train')
    tweets_test = TweetData(emb_dir,
        n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
        batchsz=args.test_batchsz, disk_load=args.disk_load, mode='train')

    for epoch in range(args.epoch // 10000):
        db = DataLoader(tweets_train, batch_size=args.task_num,
                shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            #print('pre to device has nan', torch.isnan(x_spt).any())
            x_spt = x_spt.to(device)
            y_spt = y_spt.to(device)
            x_qry = x_qry.to(device)
            y_qry = y_qry.to(device)

            #print('x spt size (batch?)', x_spt.size(), flush=True)
            #print('y spt size', y_spt.size(), flush=True)
            #print('spt labels', y_spt, flush=True)
            #print('x qry size', x_qry.size(), flush=True)
            #print('y qry size', y_qry.size(), flush=True)
            #print('qry labels', y_qry, flush=True)
            if torch.isnan(x_spt).any():
                print(x_spt)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:
                db_test = DataLoader(tweets_test, batch_size=1,
                    shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                for t_x_spt, t_y_spt, t_x_qry, t_y_qry in db_test:
                    t_x_spt = t_x_spt.squeeze(0).to(device)
                    t_y_spt = t_y_spt.squeeze(0).to(device)
                    t_x_qry = t_x_qry.squeeze(0).to(device)
                    t_y_qry = t_y_qry.squeeze(0).to(device)

                    t_accs = maml.finetunning(t_x_spt, t_y_spt, t_x_qry, t_y_qry)
                    accs_all_test.append(t_accs)

                test_acc = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Meta-test acc:', test_acc)

    num_eval_batches = 1000
    tweet_eval = TweetData(emb_dir,
        n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
        batchsz=num_eval_batches, disk_load=args.disk_load, mode='test')
    tweet_eval_db = DataLoader(tweet_eval, batch_size=1,
        shuffle=True, num_workers=1, pin_memory=True)

    final_eval_accs = []
    for e_x_spt, e_y_spt, e_x_qry, e_y_qry in tweet_eval_db:
        e_x_spt = e_x_spt.squeeze(0).to(device)
        e_y_spt = e_y_spt.squeeze(0).to(device)
        e_x_qry = e_x_qry.squeeze(0).to(device)
        e_y_qry = e_y_qry.squeeze(0).to(device)
        eval_accs = maml.finetunning(e_x_spt, e_y_spt, e_y_qry, e_y_qry)
        final_eval_accs.append(eval_accs.mean(axis=0).astype(np.float16))

    print('final evaluation')
    print('mean acc', np.mean(final_eval_accs))
    print('median acc', np.median(final_eval_accs))
    print('std acc', np.std(final_eval_accs))
    print('max acc', np.max(final_eval_accs))
    print('min acc', np.min(final_eval_accs))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--embdim', type=int, help='size of embedding vectors', default=100)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--data_dir', type=str, help='directory with embeddings', default='embs')
    argparser.add_argument('--train_batchsz', type=int, help='number of train batches', default=10000)
    argparser.add_argument('--test_batchsz', type=int, help='number of test batches', default=100)
    argparser.add_argument('--arch', type=str, help='cnn or lstm', default='cnn')
    argparser.add_argument('--disk_load', help='load embeddings from disk when needed', action='store_true')

    args = argparser.parse_args()

    main(args)
