from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import pandas as pd

from utils import utils
from utils.file_io import read_img, read_disp
from PIL import Image
import numpy as np

class CustomFullDataset(Dataset):
    def __init__(self, data_dir, dataset_name='custom_dataset_full', transform=None, save_filename=False, mode='train'):
        super(CustomFullDataset, self).__init__()

        self.data_dir = data_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.save_filename = save_filename
        self.transform = transform

        custom = {
            'train': 'filenames/custom_train.txt',
            'val': 'filenames/custom_val.txt'
        }

        custom_full = {
            'train': 'filenames/custom_train_full.txt',
            'val': 'filenames/custom_val_full.txt'
        }

        test_sim = {
            'train': 'filenames/custom_test_sim.txt',
            'test': 'filenames/custom_test_sim.txt',
        }

        test_real = {
            'train': 'filenames/custom_test_real.txt',
            'test': 'filenames/custom_test_real.txt',
        }

        dataset_name_dict = {
            'custom_dataset' : custom,
            'custom_dataset_full': custom_full,
            'custom_dataset_sim': test_sim,
            'custom_dataset_real': test_real,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]


            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None

            if(self.dataset_name == 'custom_dataset_full' or
                self.dataset_name == 'custom_dataset_sim' or
                self.dataset_name == 'custom_dataset_real'):
                meta = None if len(splits)<3 else splits[3]
                sample['meta'] = os.path.join(data_dir, meta) # new

                if (self.dataset_name == 'custom_dataset_sim' or
                self.dataset_name == 'custom_dataset_real'):
                    sample['label'] = os.path.join(data_dir, splits[4]) # label image

            self.samples.append(sample)

        # transformations TODO
        # self._augmentation()

    def _augmentation(self):
        if self.split == 'train':
            self.transformation = Compose([
                RGBShiftStereo(always_apply=True, p_asym=0.5),
                RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
            ])
        elif self.split == 'validation' or self.split == 'test' or self.split == 'validation_all':
            self.transformation = None
        else:
            raise Exception("Split not recognized")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
            sample['occ_mask'] = np.zeros_like(sample['disp']).astype(np.bool)
            sample['occ_mask_right']= np.zeros_like(sample['disp']).astype(np.bool)

        if self.transform is not None:
            sample = self.transform(sample)

        if(self.dataset_name == 'custom_dataset_full' or
            self.dataset_name == 'custom_dataset_sim' or
            self.dataset_name == 'custom_dataset_real'):
            temp = pd.read_pickle(sample_path['meta'])
            sample['intrinsic'] = temp['intrinsic']
            sample['baseline'] = abs((temp['extrinsic_l']-temp['extrinsic_r'])[0][3])

            if (self.dataset_name == 'custom_dataset_sim' or
            self.dataset_name == 'custom_dataset_real'):
                sample['label'] = np.array(Image.open(sample_path['label']))
                sample['object_ids']=temp['object_ids']
                sample['extrinsic'] = temp['extrinsic']


        return sample
