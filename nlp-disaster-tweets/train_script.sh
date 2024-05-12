#!/bin/bash

python3 script.py --data-dir './data/training_datasets/tweet' --log-identifier "train-tweet" --epochs 2 --lr 1e-5 --weight-decay 0.01 --batch-size 8 --train --weights-identifier 'train_v1.0'