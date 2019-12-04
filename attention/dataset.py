from pathlib import Path
from typing import Callable, List

import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
from transforms import tensor_transform
from utils_train import ON_KAGGLE


N_CLASSES = 1106
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
            
        maxlen = 13 #max_labels length
        #target[N_CLASSES+1] = 1
        #for cls in item.attribute_ids.split():
        #    target[int(cls)] = 1
        if len(item.attribute_ids.split()):
            culture = [int(cls) for cls in item.attribute_ids.split() if int(cls) <398]
            tag = [int(cls) for cls in item.attribute_ids.split() if int(cls)>=398]
            random.shuffle(culture)
            random.shuffle(tag)
            target = [1103] + culture + tag + [1104]
        else:
            target = [1103,1104]
        caplens = len(target)
        
        if (maxlen - len(target)):
            target = target + [1105]*(maxlen - len(target))
            
        target = torch.LongTensor(target)
        caplens = torch.LongTensor([caplens])
        return image, target, caplens


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
