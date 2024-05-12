#!/bin/bash

python3 script.py --data-dir './data/training_datasets/tweet' --log-identifier "kfolds-tweet" --epochs 3 --lr 1e-5 --weight-decay 0.01 --batch-size 8 --k-folds 