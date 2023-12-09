#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:P100
#SBATCH --partition=eng-instruction
#SBATCH --mail-user=andy2@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=12:00:00
#SBATCH --output=O-%x.%j.out

DATA_DIR=laser_embs
EMBDIM=1024

N_WAY=5
K_SPT=1
K_QRY=15
TASK_NUM=4
META_LR=0.001
UPDATE_LR=0.01
UPDATE_STEP=5
UPDATE_STEP_TEST=10

EPOCH=60000
TRAIN_BATCHSZ=10000
TEST_BATCHSZ=100

ARCH=cnn

python3 tweet_train.py --epoch ${EPOCH} \
        --n_way ${N_WAY} \
        --k_spt ${K_SPT} \
        --k_qry ${K_QRY} \
        --embdim ${EMBDIM} \
        --task_num ${TASK_NUM} \
        --meta_lr ${META_LR} \
        --update_lr ${UPDATE_LR} \
        --update_step ${UPDATE_STEP} \
        --update_step_test ${UPDATE_STEP_TEST} \
        --data_dir "${DATA_DIR}" \
        --train_batchsz ${TRAIN_BATCHSZ} \
        --test_batchsz ${TEST_BATCHSZ} \
        --arch ${ARCH} \
        --disk_load
