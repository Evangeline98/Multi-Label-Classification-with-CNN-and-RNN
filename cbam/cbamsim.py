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


class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
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

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))

# class RandomSizedCrop:
#     """Random crop the given PIL.Image to a random size
#     of the original size and and a random aspect ratio
#     of the original aspect ratio.
#     size: size of the smaller edge
#     interpolation: Default: PIL.Image.BILINEAR
#     """

#     def __init__(self, size, interpolation=Image.BICUBIC,
#                  min_aspect=4/5, max_aspect=5/4,
#                  min_area=0.25, max_area=1):
#         self.size = size
#         self.interpolation = interpolation
#         self.min_aspect = min_aspect
#         self.max_aspect = max_aspect
#         self.min_area = min_area
#         self.max_area = max_area

#     def __call__(self, img):
#         size_0 = img.size[0]
#         size_1 = img.size[1]
#         print(size_0, size_1)
#         img_data = np.array(img)
#         if ((size_0/size_1>=1.3) or (size_1/size_0>=1.3)):
#             w_resized = int(img.size[0] * 300 / img.size[1])
#             h_resized = int(img.size[1] * 300 / img.size[0])
#             if size_0 < size_1:
#                 resized = img.resize((w_resized ,300))
#                 pad_width = 300 - w_resized
#                 df = pd.DataFrame(img_data[0,:,:])
#                 padding = (pad_width // 2, 0, pad_width-(pad_width//2), 0)
#             else:
#                 resized = img.resize((300, h_resized))
#                 pad_height = 300 - h_resized
#                 df = pd.DataFrame(img_data[:,0,:])
#                 padding = (0, pad_height // 2, 0, pad_height-(pad_height//2))
            
#             AvgColour = tuple([int(i) for i in df.mean()])
#             resized_w_pad = ImageOps.expand(resized, padding, fill=AvgColour)
#         else:
#             for attempt in range(10):
#                 print(attempt)
#                 area = img.size[0] * img.size[1]
#                 target_area = random.uniform(self.min_area, self.max_area) * area
#                 aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)
    
#                 w = int(round(math.sqrt(target_area * aspect_ratio)))
#                 h = int(round(math.sqrt(target_area / aspect_ratio)))

#                 if random.random() < 0.5:
#                     w, h = h, w
    
#                 if w <= img.size[0] and h <= img.size[1]:
#                     x1 = random.randint(0, img.size[0] - w)
#                     y1 = random.randint(0, img.size[1] - h)
    
#                     img = img.crop((x1, y1, x1 + w, y1 + h))
#                     assert(img.size == (w, h))
    
#                     return img.resize((self.size, self.size), self.interpolation)
                
#                 scale = Resize(self.size, interpolation=self.interpolation)
#                 crop = CenterCrop(self.size)
#                 resized_w_pad = crop(scale(img))
#         # Fallback
#         return resized_w_pad

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
DATA_ROOT = Path('../input/imet-2019-fgvc6' if ON_KAGGLE else './data')


class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame,count: pd.DataFrame,thres,
                 image_transform: Callable, debug: bool = True):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._debug = debug
        self.index = np.where(count['count'] < thres)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_transform_image(
            item, self._root, self._image_transform, debug=self._debug)
        target = torch.zeros(N_CLASSES)
        for cls in item.attribute_ids.split():
            target[int(cls)] = 1
        target[self.index] = 0
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
 
 
 ####################################model#################################
import math
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.init as init
import os


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            eps=bn_eps)
        if self.activate:
            assert (activation is not None)
            if isfunction(activation):
                self.activ = activation()
            elif isinstance(activation, str):
                if activation == "relu":
                    self.activ = nn.ReLU(inplace=True)
                elif activation == "relu6":
                    self.activ = nn.ReLU6(inplace=True)
                else:
                    raise NotImplementedError()
            else:
                self.activ = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    1x1 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation,
        activate=activate)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    3x3 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation,
        activate=activate)


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=3,
                  bias=False,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    7x7 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        activation=activation,
        activate=activate)
        
class ResBlock(nn.Module):
    """
    Simple ResNet block for residual path in ResNet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(nn.Module):
    """
    ResNet bottleneck block for residual path in ResNet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 conv1_stride=False,
                 bottleneck_factor=4):
        super(ResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=(stride if conv1_stride else 1))
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=(1 if conv1_stride else stride),
            padding=padding,
            dilation=dilation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
class ResInitBlock(nn.Module):
    """
    ResNet specific initial block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResInitBlock, self).__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class MLP(nn.Module):
    """
    Multilayer perceptron block.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Linear(
            in_features=channels,
            out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=mid_channels,
            out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(nn.Module):
    """
    CBAM channel gate block.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(ChannelGate, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * att
        return x


class SpatialGate(nn.Module):
    """
    CBAM spatial gate block.
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = conv7x7_block(
            in_channels=2,
            out_channels=1,
            activate=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att1 = x.max(dim=1)[0].unsqueeze(1)
        att2 = x.mean(dim=1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.conv(att)
        att = self.sigmoid(att)
        x = x * att
        return x


class CbamBlock(nn.Module):
    """
    CBAM attention block for CBAM-ResNet.
    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(CbamBlock, self).__init__()
        self.ch_gate = ChannelGate(
            channels=channels,
            reduction_ratio=reduction_ratio)
        self.sp_gate = SpatialGate()

    def forward(self, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x


class CbamResUnit(nn.Module):
    """
    CBAM-ResNet unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(CbamResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=False)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
        self.cbam = CbamBlock(channels=out_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.cbam(x)
        x = x + identity
        x = self.activ(x)
        return x


class CbamResNet(nn.Module):
    """
    CBAM-ResNet model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(CbamResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), CbamResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create CBAM-ResNet model with specific parameters.
    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported CBAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CbamResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def cbam_resnet18(**kwargs):
    """
    CBAM-ResNet-18 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="cbam_resnet18", **kwargs)


def cbam_resnet34(**kwargs):
    """
    CBAM-ResNet-34 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="cbam_resnet34", **kwargs)


def cbam_resnet50(**kwargs):
    """
    CBAM-ResNet-50 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="cbam_resnet50", **kwargs)


def cbam_resnet101(**kwargs):
    """
    CBAM-ResNet-101 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="cbam_resnet101", **kwargs)


def cbam_resnet152(**kwargs):
    """
    CBAM-ResNet-152 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="cbam_resnet152", **kwargs)#


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
from torch.optim import Adam
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
            outputs = torch.sigmoid(model(inputs))
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, criterion, *, params,folds, count,
          init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=2) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model.pt'
    best_model_path = run_root / 'best-model.pt'
    pretrain_path = Path('../input/modelcbam')/'best-model (1).pt'
    if pretrain_path.exists():
        state = load_model(model, pretrain_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = 50
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
    
    def make_loader(df: pd.DataFrame, image_transform, count: pd.DataFrame, thres) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df,  count, thres ,image_transform,  debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    train_loader = make_loader(train_fold, train_transform, count, args.count)
    valid_loader = make_loader(valid_fold, test_transform,count,0)
    
   ##############
    validation(model, criterion, valid_loader, use_cuda)
    validation2(model, criterion, valid_loader, use_cuda)
       
       
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        losses = []

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, lr {lr}')
          
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
    # C is the number of classes.
                batch_size = inputs.size(0)
                #smoothed_labels =0.9*targets + 0.1*(torch.ones((batch_size,N_CLASSES)).cuda()-targets)
                #smoothed_labels = smoothed_labels.cuda()
                outputs = model(inputs)

                loss = _reduce_loss(criterion(outputs, targets))
                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, use_cuda)
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
                lr /= 5
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
            outputs = model(inputs)
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
    argsorted = all_predictions.argsort(axis=1)
    for threshold in [0.05,0.10, 0.15, 0.20]:
        metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
            binarize_prediction(all_predictions, threshold, argsorted))
    metrics['valid_loss'] = np.mean(all_losses)
    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
        metrics.items(), key=lambda kv: -kv[1])))

    return metrics

# def validation(
#         model: nn.Module, criterion, valid_loader, use_cuda,
#         ) -> Dict[str, float]:
#     model.eval()
#     all_losses, all_predictions, all_targets = [], [], []
#     with torch.no_grad():
#         for inputs, targets in valid_loader:
#             all_targets.append(targets.numpy().copy())
#             if use_cuda:
#                 inputs, targets = inputs.cuda(), targets.cuda()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             all_losses.append(_reduce_loss(loss).item())
#             predictions = torch.sigmoid(outputs)
#             res_argsorted = predictions.cpu().numpy().argsort()
#             threshold = 0.1
#             result = binarize_prediction(predictions.cpu().numpy(), threshold, res_argsorted)
#             #    indexes = np.arange(N_CLASSES) + 1
#             for i in range(result.shape[0]):
#                 pred_labels = np.nonzero(result[i,])
#                 pred_origin_labels = np.nonzero(targets.cpu().numpy()[i,])
#                 print("prediction", i, pred_labels)
#                 print("original" , i, pred_origin_labels)
#                 pred_prob = [j for j in predictions.cpu().numpy()[i,]]
#                 f = open('PredLabels.csv','a')
#                 f.write('Pred %d: ' % i)
#                 for s in [j for j in pred_labels]:
#                     f.write('%s,' % s)
#                     b = s
#                 f.write("\n")
#                 f.write('PrdP %d: ' % i)
#                 pred_prob_index = []
#                 for ll in b:
#                     pred_prob_index.append(pred_prob[ll])
#                 for j in pred_prob_index:
#                     f.write('%s,' % j)
# #                f.write('%s,' % pred_prob[pred_labels])
#                 f.write("\n")
#                 f.write('Orig %d: ' % i)
#                 for x in [j for j in pred_origin_labels]:
#                     f.write('%s,' % x)
#                     b = x
#                 f.write("\n")
#                 f.write('OrgP %d: ' % i)
#                 orig_prob_index = []
#                 for ll in b:
#                     orig_prob_index.append(pred_prob[ll])
#                 for j in orig_prob_index:
#                     f.write('%s,' % j)
#                 f.write("\n")
#                 f.close()
#             all_predictions.append(predictions.cpu().numpy())
#     all_predictions = np.concatenate(all_predictions)
#     all_targets = np.concatenate(all_targets)

#     def get_score(y_pred):
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', category=UndefinedMetricWarning)
#             return fbeta_score(
#                 all_targets, y_pred, beta=2, average='samples')

#     metrics = {}
#     argsorted = all_predictions.argsort(axis=1)
#     threshold = 0.1
# #    for threshold in [0.05,0.10, 0.15, 0.20]:
#     metrics[f'valid_f2_th_{threshold:.2f}'] = get_score(
#             binarize_prediction(all_predictions,threshold, argsorted))

#     metrics['valid_loss'] = np.mean(all_losses)
#     print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(
#         metrics.items(), key=lambda kv: -kv[1])))

#     return metrics
    
def validation2(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
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
    for threshold in [0.05,0.10, 0.15, 0.20,0.25]:
        for t in [0.1,0.2,0.3,0.4]:
            metrics[f'valid_f2_th_{threshold:.2f}{t:.2f}'] = get_score(
                    np.hstack([binarize_prediction(all_predictions[:,:398], t, argsorted1,max_labels = 1),
                        binarize_prediction(all_predictions[:,398:],threshold,argsorted2,max_labels = 11)]))
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
        self.run_root = 'model'
        self.batch_size = 32
        self.step  = 1
        self.workers = 2
        self.lr = 1e-4
        self.patience = 4
        self.clean = 0
        self.n_epochs = 25
        self.tta = 4
        self.debug = 'store_true'
        self.pretrained = 0
        self.threshold = 0.1
        self.folds = 100
        self.count = 0
        
args = arg()

run_root = Path(args.run_root)
if run_root.exists() and args.clean:
    shutil.rmtree(run_root)
run_root.mkdir(exist_ok=True, parents=True)
(run_root / 'params.json').write_text(
    json.dumps(vars(args), indent=4, sort_keys=True))
    
    
folds = make_folds(n_folds = args.folds)
train_root = DATA_ROOT / 'train'
        
# import numpy as np     
# def process_glove_line(line, dim):      
#     word = None
#     embedding = None

#     #try:
#     splitLine = line.split()
#     word = " ".join(splitLine[:len(splitLine)-dim])
#     embedding = np.array([float(val) for val in splitLine[-dim:]])
#   # except:
#   #     print(line)

#     return word, embedding

# def load_glove_model(glove_filepath, dim):
#     with open(glove_filepath, encoding="utf8" ) as f:
#         content = f.readlines()
#         model = {}
#         for line in content:
#             word, embedding = process_glove_line(line, dim)
#             if embedding is not None:
#                 model[word] = embedding
#         return model
        


# from torch.nn.functional import cosine_similarity
# vectors = load_glove_model("../input/glove840b300dtxt/glove.840B.300d.txt", 300)
# vectors[None] = np.zeros(300)

# def EmbeddingPar():
#     data = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")
#     name = data['attribute_name'].str.split("::",expand = True)
#     name1 = name[1].str.split(expand = True)
#     name = pd.DataFrame(np.concatenate([name,name1],axis=1))
#     embedding = torch.zeros(N_CLASSES,300)
#     print(name.shape[0])
#     for i in range(name.shape[0]):
#         emb = np.zeros((5,300))
#         for j in range(5):
#             try:
#                 emb[j] = vectors[name.iloc[i,j]]
#             except:
#                 emb[j] = np.zeros(300)
#         emb = torch.Tensor(emb)
#         embedding[i] = torch.sum(emb,dim = 0)
#         n = np.sum(1-pd.isnull(name.iloc[i]))
#         embedding[i]/=n
        
#     Sim = torch.zeros(N_CLASSES,N_CLASSES)
#     for i in range( N_CLASSES):
#         for j in range( N_CLASSES):
#             if (i>=398 and j>=398) or (i<398 and j<398):
#                 if i!=j:
#                     Sim[i,j] = cosine_similarity(embedding[i,:].view(1,-1),embedding[j,:].view(1,-1))

#     return embedding,Sim

# embed,Sim = EmbeddingPar()
# torch.save(Sim,run_root/'Sim.pt')
#on gpu you should torch.load('Sim.pt')
Sim = torch.load('../input/modelsim/Sim.pt')
Sim = Sim*torch.FloatTensor((Sim>0.5).numpy())
Sim = Sim.cuda()

class SimilarityLoss(nn.Module):
    def __init__(self, sim):
        '''
        sim : N_class*N_class
        '''
        super(SimilarityLoss, self).__init__()
        self.sim = sim
        
    def forward(self,input,target):
        input1 = torch.sigmoid(input.clone())
        Smatrix = torch.matmul(input1, self.sim)+1
        #print(Smatrix)
        P = torch.exp(input)
        #print(P)
        #print(Smatrix)
        loss = -(Smatrix*target*(input-torch.log(P+1))+(1-target)*(-torch.log(1+P))) 
        return loss
        
        
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
        loss = -(Smatrix*target*(input-torch.log(P+1))+(1-target)*(-torch.log(1+P))) 

        return loss
        
        
    
criterion = SimilarityLoss1(Sim).cuda()

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])
        
class Net(nn.Module):
    def __init__(self, num_classes, dropout=True):
        super().__init__()
        self.net = cbam_resnet50()
        self.net.load_state_dict(torch.load('../input/cbam-resnet50/cbam_resnet50.pth'))

        #self.net = nn.Sequential(*list(model0.children())[0])
       # print(self.net.output)
        if dropout:
           # model.add_module('fc', torch.nn.Linear(4096, out_num)) 
            self.net.output = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.output.in_features, num_classes)
                )
        else:
            self.net.output = nn.Linear(self.net.output.in_features, num_classes)
        #self.finetune()

    def forward(self, x):
        return self.net(x)
        
    def finetune(self):
        for para in list(self.net.parameters())[:-2]:
            para.requires_grad=False 
        
model = Net(N_CLASSES)
use_cuda = cuda.is_available()
print(use_cuda)
#fresh_params = list(model.fresh_params())
all_params = list(model.parameters())
if use_cuda:
    model = model.cuda()


    
from collections import Counter
def get_count():
	df = pd.read_csv('../input/imet-2019-fgvc6/train.csv' if ON_KAGGLE else '/nfsshare/home/white-hearted-orange/data/train.csv')
	cls_counts = Counter(cls for classes in df['attribute_ids'].str.split() for cls in classes)
	stat = cls_counts.most_common()
	stat1 = pd.DataFrame(stat)
	stat1.columns=('attribute_id','count')
	stat1['attribute_id'].astype('int')
	return stat1
count = get_count()
	
train_kwargs = dict(
    args= args,
    model = model,
    folds = folds,
    count = count,
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




