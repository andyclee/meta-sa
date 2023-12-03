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
    config_cnn = [
        # (layer_name, [params]),
        ('conv1d', [32, 1, 3, 1, 'same']),
        ('elu', [True]),
        ('conv1d', [32, 32, 3, 1, 'same']),
        ('elu', [True]),
        ('conv1d', [32, 32, 3, 1, 'same']),
        ('relu', [True]),
        ('globalmax_pool1d', [True]),
        ('dense', [128, 32]),
        ('relu', [True]),
        ('dropout', [0.3, True]),
        ('dense', [3, 128]),
        ('softmax', [])
    ]

    config_lstm = [

    ]

    config = config_cnn

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
        batchsz=10000)
    tweets_test = TweetData(emb_dir,
        n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
        batchsz=100)

    for epoch in range(args.epoch // 10000):
        db = DataLoader(tweets_train, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt = x_spt.to(device)
            y_spt = y_spt.to(device)
            x_qry = x_qry.to(device)
            y_qry = y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:
                db_test = DataLoader(tweets_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                for t_x_spt, t_y_spt, t_x_qry, t_y_qry in db_test:
                    t_x_spt = t_x_spt.squeeze(0).to(device)
                    t_y_spt = t_y_spt.squeeze(0).to(device)
                    t_x_qry = t_x_qry.squeeze(0).to(device)
                    t_y_qry = t_y_qry.squeeze(0).to(device)

                    t_accs = maml.finetunning(t_x_spt, t_y_spt, t_x_qry, t_y_qry)
                    accs_all_test.append(t_accs)

                test_acc = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', test_acc)

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

    args = argparser.parse_args()

    main(args)
