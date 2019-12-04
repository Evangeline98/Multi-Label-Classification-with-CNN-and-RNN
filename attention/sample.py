import pandas as pd
from dataset import TrainDataset, DATA_ROOT
from transforms import train_transform, test_transform
from torch.utils.data import DataLoader
from pathlib import Path
from utils_train import ON_KAGGLE



def sampler(size, batch):
    '''
    size: number of samples in train and validation set
    '''
    folds = pd.read_csv(Path('../input/attention/folds.csv' if ON_KAGGLE else 'folds.csv'))
    train_root = DATA_ROOT / ('train')
    train_fold = folds[folds['fold'] != 0]
    valid_fold = folds[folds['fold'] == 0]
    
    if size:
        train_fold = train_fold[:size]
        valid_fold = valid_fold[:size]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        data =  DataLoader(
            TrainDataset(train_root, df, image_transform),
            batch_size = batch,
            shuffle=True,
            num_workers=2,
        )
        #print(data.batch_size)
        return data


    train_loader = make_loader(train_fold, train_transform)
    valid_loader = make_loader(valid_fold, test_transform)
    return train_loader,  valid_loader

if __name__ == '__main__':
    train, valid = sampler(100,10)
    for i ,(input,targets, caplens) in enumerate(train):
        if i == 0:
#            print(input.numpy().shape)
#            print(input)
            print(caplens)
#            print(targets)
            break
