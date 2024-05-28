import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
import json


def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False,
                dataset='CASIAB') -> None:

    sinfo = img_groups[0]  
    img_paths = list(img_groups[1]) 
    to_pickle = []  
    for img_file in sorted(img_paths):
        if verbose:  
            print(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {img_file}')

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        if img.sum() <= 10000:
            if verbose:
                print(f'Image sum: {img.sum()}')
            print(f'{img_file} has no data.')
            continue

          y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top: y_btm + 1, :]

        ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        x_csum = img.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img.sum() / 2:
                x_center = idx
                break

        if not x_center:
            print(f'{img_file} has no center.')
            continue

        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _ = np.zeros((img.shape[0], half_width))
            img = np.concatenate([_, img, _], axis=1)

        img = img[:, left: right].astype('uint8')

        to_pickle.append(img)

    if to_pickle:
        to_pickle = np.asarray(to_pickle)  
        dst_path = os.path.join(output_path, *sinfo)  
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            print(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))
        print(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')

    if len(to_pickle) < 5:
        print(f'{sinfo} has less than 5 valid data.')



def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False,
             dataset: str = 'CASIAB') -> None:

    img_groups = defaultdict(list)
    print(f'Listing {input_path}')
    total_files = 0
    for img_path in input_path.rglob('*.png'):
        if 'gei.png' in img_path.as_posix():
            continue
        if verbose:
            print(f'Adding {img_path}')
         *_, sid, seq, view, _ = img_path.as_posix().split('/')

        img_groups[(sid, seq, view)].append(img_path)
        total_files += 1 

    print(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')


    with mp.Pool(workers) as pool:
        print(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(
                partial(imgs2pickle, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset),
                img_groups.items()):
            progress.update(1)
    print('Done')


if __name__ == '__main__':
    # input_path = Path('')
    # output_path = Path('')
    input_path = Path('')
    output_path = Path('')
    pretreat(input_path=input_path,output_path=output_path)
