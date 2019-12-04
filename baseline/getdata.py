import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam
import tqdm
import torchvision

import models
from dataset import TrainDataset, TTADataset, get_ids, N_CLASSES, DATA_ROOT
from transforms import train_transform, test_transform
from utils import (
    write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
    ON_KAGGLE)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test'])
    arg('run_root')
    #arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    #arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    #arg('--lr', type=float, default=1e-4)
    #arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    #arg('--n-epochs', type=int, default=100)
    #arg('--epoch-size', type=int)
    #arg('--tta', type=int, default=4)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / ('train_sample' if args.use_sample else 'train')
    if args.use_sample:
        folds = folds[folds['Id'].isin(set(get_ids(train_root)))]
    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(train_fold, train_transform)
        valid_loader = make_loader(valid_fold, test_transform)
        #for i, (inputs, targets) in enumerate(train_loader.dataset):
        #    print(i," ",targets)

        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')


if __name__ == '__main__':
    main()
