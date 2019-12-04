from datetime import datetime
import json
import glob
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
import torch
from torch import nn
from torch.utils.data import DataLoader


ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ


def gmean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).agg(lambda x: gmean(list(x)))


def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()


def load_model(model: nn.Module, path: Path) -> Dict:
    state = torch.load(str(path))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state


class ThreadingDataLoader(DataLoader):
    def __iter__(self):
        sample_iter = iter(self.batch_sampler)
        if self.num_workers == 0:
            for indices in sample_iter:
                yield self.collate_fn([self._get_item(i) for i in indices])
        else:
            prefetch = 1
            with ThreadPool(processes=self.num_workers) as pool:
                futures = []
                for indices in sample_iter:
                    futures.append([pool.apply_async(self._get_item, args=(i,))
                                    for i in indices])
                    if len(futures) > prefetch:
                        yield self.collate_fn([f.get() for f in futures.pop(0)])
                    # items = pool.map(lambda i: self.dataset[i], indices)
                    # yield self.collate_fn(items)
                for batch_futures in futures:
                    yield self.collate_fn([f.get() for f in batch_futures])

    def _get_item(self, i):
        return self.dataset[i]


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()



def _smooth(ys, indices):
    return [np.mean(ys[idx: indices[i + 1]])
            for i, idx in enumerate(indices[:-1])]


import random
import math

from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip)


# class RandomSizedCrop:
#     """Random crop the given PIL.Image to a random size
#     of the original size and and a random aspect ratio
#     of the original aspect ratio.
#     size: size of the smaller edge
#     interpolation: Default: PIL.Image.BILINEAR
#     """

#     def __init__(self, size, interpolation=Image.BILINEAR,
#                  min_aspect=4/5, max_aspect=5/4,
#                  min_area=0.25, max_area=1):
#         self.size = size
#         self.interpolation = interpolation
#         self.min_aspect = min_aspect
#         self.max_aspect = max_aspect
#         self.min_area = min_area
#         self.max_area = max_area

#     def __call__(self, img):
#         for attempt in range(10):
#             area = img.size[0] * img.size[1]
#             target_area = random.uniform(self.min_area, self.max_area) * area
#             aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if random.random() < 0.5:
#                 w, h = h, w

#             if w <= img.size[0] and h <= img.size[1]:
#                 x1 = random.randint(0, img.size[0] - w)
#                 y1 = random.randint(0, img.size[1] - h)

#                 img = img.crop((x1, y1, x1 + w, y1 + h))
#                 assert(img.size == (w, h))

#                 return img.resize((self.size, self.size), self.interpolation)

#         # Fallback
#         scale = Resize(self.size, interpolation=self.interpolation)
#         crop = CenterCrop(self.size)
#         return crop(scale(img))

class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        size_0 = img.size[0]
        size_1 = img.size[1]
        print(size_0, size_1)
        img_data = np.array(img)
        if ((size_0/size_1>=1.3) or (size_1/size_0>=1.3)):
            w_resized = int(img.size[0] * 300 / img.size[1])
            h_resized = int(img.size[1] * 300 / img.size[0])
            if size_0 < size_1:
                resized = img.resize((w_resized ,300))
                pad_width = 300 - w_resized
                df = pd.DataFrame(img_data[0,:,:])
                padding = (pad_width // 2, 0, pad_width-(pad_width//2), 0)
            else:
                resized = img.resize((300, h_resized))
                pad_height = 300 - h_resized
                df = pd.DataFrame(img_data[:,0,:])
                padding = (0, pad_height // 2, 0, pad_height-(pad_height//2))
            
            AvgColour = tuple([int(i) for i in df.mean()])
            resized_w_pad = ImageOps.expand(resized, padding, fill=AvgColour)
            
           # plt.figure(figsize=(8,8))
           # plt.subplot(133)
           # plt.imshow(resized_w_pad)
           # plt.axis('off')
           # plt.title('Padded Image',fontsize=15)
           ## 
           # plt.show()
        else:
            for attempt in range(10):
                print(attempt)
                area = img.size[0] * img.size[1]
                target_area = random.uniform(self.min_area, self.max_area) * area
                aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)
    
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if random.random() < 0.5:
                    w, h = h, w
    
                if w <= img.size[0] and h <= img.size[1]:
                    x1 = random.randint(0, img.size[0] - w)
                    y1 = random.randint(0, img.size[1] - h)
    
                    img = img.crop((x1, y1, x1 + w, y1 + h))
                    assert(img.size == (w, h))
    
                    return img.resize((self.size, self.size), self.interpolation)
                
                scale = Resize(self.size, interpolation=self.interpolation)
                crop = CenterCrop(self.size)
                resized_w_pad = crop(scale(img))
        # Fallback
        return resized_w_pad


train_transform = Compose([
    RandomCrop(224),
    RandomHorizontalFlip(),
])


test_transform = Compose([
    RandomCrop(224),
    RandomHorizontalFlip(),
])


tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


N_CLASSES = 1103
DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else '/nfsshare/home/white-hearted-orange/data')


class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        target = torch.zeros(N_CLASSES)
        for cls in item.attribute_ids.split():
            target[int(cls)] = 1
        return image, target


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame,
                 image_transform: Callable, tta: int):
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._tta = tta

    def __len__(self):
        return len(self._df) * self._tta

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_transform_image(item, self._root, self._image_transform)
        return image, item.id


def load_transform_image(
        item, root: Path, image_transform: Callable, debug: bool = False):
    image = load_image(item, root)
    image = image_transform(image)
    if debug:
        image.save('_debug.png')
    return tensor_transform(image)


def load_image(item, root: Path) -> Image.Image:
    image = cv2.imread(str(root / f'{item.id}.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.png')})
    
import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
import tqdm


def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df
 ###########################models###############################
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M



class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
        
    else:
        #net = net_cls(pretrained=pretrained)
        net = net_cls()
        model_name = net_cls.__name__
        net.load_state_dict(torch.load(f'/nfsshare/home/white-hearted-orange/kaggle-imet-2019-master/imet/{model_name}.pth'))
        print(model_name)

    return net

class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet101, dropout=True):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool() 
        n = self.net.fc.in_features
        #print(n)
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        if dropout:
            self.culture = nn.Sequential(
                nn.Dropout(),
                nn.Linear(n, 398)
                )
            self.hidden = nn.Linear(398,50)
            self.relu = nn.ReLU()
            self.tag = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(n+50,num_classes-398)
                    )
        else:
            self.culture = nn.Linear(n, 398)
            self.hidden = nn.Linear(398, 50)
            self.relu = nn.ReLU()
            self.tag = nn.Linear(n + 50, num_classes - 398)

    def forward(self, x):
        h0 = self.net(x)
        h0 = h0.view(h0.size()[:2])
        #print(h0.size())
        h1 = self.culture(h0)
        h2 =self.relu( self.hidden(h1))
        #print(h2.size())
        h3 = self.tag(torch.cat([h0,h2],1))
        prediction = torch.cat([h1,h3],1)
        return h1,h3,prediction



resnet50 = partial(ResNet, net_cls = M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)

########################main.py########################################################
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
from torch.optim import Adam,SGD, lr_scheduler
import tqdm


def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta: int, workers: int, use_cuda: bool):
    loader = DataLoader(
        dataset=TTADataset(root, df, test_transform, tta=tta),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
            if use_cuda:
                inputs = inputs.cuda() 
            _,_, out = model(inputs)
            outputs = torch.sigmoid(out)
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, criterion, *, params,folds,
          init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=2) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model.pt'
    best_model_path = run_root / 'best-model.pt'
   # pretrain_path = Path('../input/model1')/'best-model.pt'
   
    if  best_model_path.exists():
        state = load_model(model,best_model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    
    
    ### doing cv
    train_fold = folds[folds['fold'] != 0]
    valid_fold = folds[folds['fold'] == 0]
    
   

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    train_loader = make_loader(train_fold, train_transform)
    valid_loader = make_loader(valid_fold, test_transform)
   ##############
    #validation(model, criterion, valid_loader, use_cuda)  
    #validation2(model, criterion, valid_loader, use_cuda)  
    
    cultureloss = nn.BCEWithLogitsLoss(reduction = 'none',pos_weight = torch.ones([398]))
    tagloss = nn.BCEWithLogitsLoss(reduction='none',pos_weight = torch.ones([705]))
    cultureloss = cultureloss.cuda()
    tagloss = tagloss.cuda()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)  
    for epoch in range(epoch, n_epochs + 1):
        scheduler.step()
        model.train()
        losses = []
        
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, best_lr {lr}')
          
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                #label smoothing
                batch_size = inputs.size(0)
                #smoothed_labels =0.9*targets + 0.1*(torch.ones((batch_size,N_CLASSES)).cuda()-targets)
                #smoothed_labels = smoothed_labels.cuda()
                h1, h3,outputs = model(inputs)
                

                loss = 10*_reduce_loss(cultureloss(h1, targets[:,:398])) + _reduce_loss(tagloss(h3, targets[:,398:])) +  _reduce_loss(criterion(outputs, targets))
           
                batch_size = inputs.size(0)
                totalloss = _reduce_loss(criterion(outputs, targets))
                
                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(totalloss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, use_cuda)
            validationp(model, criterion, valid_loader, use_cuda)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr_changes +=1
                if lr_changes > max_lr_changes:
                    break
                lr *= 1
                print(f'lr updated to {lr}')
                lr_reset_epoch = epoch
                optimizer = init_optimizer(params, lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return False
    return True


def validation(
         model: nn.Module, criterion, valid_loader, use_cuda,
         ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            _,_,outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(_reduce_loss(loss).item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted = all_predictions.argsort(axis = 1)
    for threshold in [0.10, 0.15, 0.20,0.25,0.30]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
                binarize_prediction(all_predictions, threshold, argsorted))
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics

def validationp(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            _,_,outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss0 = nn.BCEWithLogitsLoss(reduction='none')(outputs,targets)
            all_losses.append(_reduce_loss(loss).item())
            #print(_reduce_loss(loss).item())
            #print(_reduce_loss(loss0).item())
            predictions = torch.sigmoid(outputs)
            res_argsorted = predictions.cpu().numpy().argsort()
            threshold = 0.1
            result = binarize_prediction(predictions.cpu().numpy(), threshold, res_argsorted)
            #    indexes = np.arange(N_CLASSES) + 1
            for i in range(result.shape[0]):
                pred_labels = np.nonzero(result[i,])
                pred_origin_labels = np.nonzero(targets.cpu().numpy()[i,])
                #print("prediction", i, pred_labels)
                #print("original" , i, pred_origin_labels)
                pred_prob = [j for j in predictions.cpu().numpy()[i,]]
                f = open('PredLabels.csv','a')
                f.write('Pred %d: ' % i)
                for s in [j for j in pred_labels]:
                    f.write('%s,' % s)
                    b = s
                f.write("\n")
                f.write('PrdP %d: ' % i)
                pred_prob_index = []
                for ll in b:
                    pred_prob_index.append(pred_prob[ll])
                for j in pred_prob_index:
                    f.write('%s,' % j)
#                f.write('%s,' % pred_prob[pred_labels])
                f.write("\n")
                f.write('Orig %d: ' % i)
                for x in [j for j in pred_origin_labels]:
                    f.write('%s,' % x)
                    b = x
                f.write("\n")
                f.write('OrgP %d: ' % i)
                orig_prob_index = []
                for ll in b:
                    orig_prob_index.append(pred_prob[ll])
                for j in orig_prob_index:
                    f.write('%s,' % j)
                f.write("\n")
                f.close()
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    threshold = 0.1
#    for threshold in [0.05,0.10, 0.15, 0.20]:
    metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(all_predictions,threshold, argsorted))

    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics

    
def validation2(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    ell_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            _,_,outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(_reduce_loss(loss).item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return fbeta_score(
                all_targets, y_pred, beta=2, average='samples')

    metrics = {}
    argsorted1 = all_predictions[:,:398].argsort(axis = 1)
    argsorted2 = all_predictions[:,398:].argsort(axis = 1)
    for threshold in [0.05,0.10, 0.15, 0.20,0.25,0.3]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
                np.hstack([binarize_prediction(all_predictions[:,:398], 0.3, argsorted1),
                    binarize_prediction(all_predictions[:,398:],threshold,argsorted2)]))
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    #assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


class arg():
    def __init__(self):
        self.run_root = '/nfsshare/home/white-hearted-orange/kaggle-imet-2019-master/imet/rethink1/model2'
        self.model = 'resnet101'
        self.batch_size = 64
        self.step  = 1
        self.workers = 2
        self.lr = 1e-4
        self.patience = 2
        self.clean = 0
        self.n_epochs = 35
        self.tta = 4
        self.debug = 'store_true'
        self.pretrained = 0
        self.threshold = 0.1
        self.folds = 5
        
args = arg()

run_root = Path(args.run_root)
folds = make_folds(n_folds = args.folds)
train_root = DATA_ROOT / 'train'

Sim = torch.load('/nfsshare/home/white-hearted-orange/data/Sim.pt')
Sim = Sim.cuda()

# class SimilarityLoss(nn.Module):
#     def __init__(self, sim):
#         '''
#         sim : N_class*N_class
#         '''
#         super(SimilarityLoss, self).__init__()
#         self.sim = sim
#
#     def forward(self,input,target):
#         input1 = torch.sigmoid(input.clone())
#         Smatrix = torch.matmul(input1, self.sim)+1
#         #print(Smatrix)
#         P = torch.sigmoid(input)
#         loss = -(Smatrix*target*torch.log(P)+(1-target)*(torch.log(1-P)))
#         return loss
        
        
class SimilarityLoss1(nn.Module):
    def __init__(self, sim):
        '''
        sim : N_class*N_class
        '''
        super(SimilarityLoss1, self).__init__()
        self.sim = sim
        
    def forward(self,input,target):
        Smatrix =  torch.matmul(target, self.sim) + 1
        #print(Smatrix)
        P = torch.exp(input)
        #print(P)
        #print(Smatrix)
        loss = -(Smatrix*target*(input-torch.log(P+1))+(1-target)*(-torch.log(1+P))) 
       
        return loss
        
criterion = SimilarityLoss1(Sim).cuda()


model = ResNet(num_classes=N_CLASSES, pretrained=args.pretrained)
use_cuda = cuda.is_available()
print(use_cuda)
all_params = list(model.parameters())
if use_cuda:
    model = model.cuda()


if run_root.exists() and args.clean:
    shutil.rmtree(run_root)
run_root.mkdir(exist_ok=True, parents=True)
(run_root / 'params.json').write_text(
    json.dumps(vars(args), indent=4, sort_keys=True))
print((run_root/'model.pt').exists())
train_kwargs = dict(
    args= args,
    model = model,
    folds = folds,
    criterion=criterion,
    patience=args.patience,
    init_optimizer=lambda params, lr: Adam(params, lr),
    use_cuda=use_cuda,
)

train(params=all_params, **train_kwargs)


load_model(model, run_root/'best-model.pt')
predict_kwargs = dict(
    batch_size=args.batch_size,
    tta=args.tta,
    use_cuda=use_cuda,
    workers=args.workers,
)

   
test_root = DATA_ROOT / ('test')
ss = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
predict(model, df=ss, root=test_root,
        out_path=run_root / 'test.h5',
        **predict_kwargs)
        
def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)

sample_submission = pd.read_csv(
        DATA_ROOT / 'sample_submission.csv', index_col='id')   
df = pd.read_hdf(run_root / 'test.h5', index_col='id')
df = df.reindex(sample_submission.index)
df = mean_df(df)
df[:] = binarize_prediction(df.values, threshold=args.threshold)
df = df.apply(get_classes, axis=1)
df.name = 'attribute_ids'
df.to_csv('submission.csv', header=True)
