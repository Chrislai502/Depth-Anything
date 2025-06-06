# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import itertools
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

from zoedepth.utils.config import change_dataset

from .ddad import get_ddad_loader
from .diml_indoor_test import get_diml_indoor_loader
from .diml_outdoor_test import get_diml_outdoor_loader
from .diode import get_diode_loader
from .hypersim import get_hypersim_loader
from .ibims import get_ibims_loader
from .sun_rgbd_loader import get_sunrgbd_loader
from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader
# from .art import get_art_loader

import sys
current_file_path = os.path.abspath(__file__)  # Absolute path to train_mix_infer.py
root_dir = os.path.abspath(os.path.join(current_file_path, '..', '..', '..', '..'))  # Go two levels up
sys.path.append(root_dir)  # Add the root directory to sys.path
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from .preprocess import CropParams, get_white_border, get_black_border
import matplotlib.pyplot as plt


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, config = None, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, config=config, **kwargs)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        
        img_size = img_size if self.config.get("do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size, config=config)

        if mode == 'train':

            self.training_samples = DataLoadPreprocess(config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'eval':
            self.testing_samples = DataLoadPreprocess(config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=kwargs.get("shuffle", False), num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def repetitive_roundrobin(self, *iterables):
        """
        cycles through iterables but sample wise
        first yield first sample from first iterable then first sample from second iterable and so on
        then second sample from first iterable then second sample from second iterable and so on

        If one iterable is shorter than the others, it is repeated until all iterables are exhausted
        repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
        """
        # Repetitive roundrobin
        iterables_ = [iter(it) for it in iterables]
        exhausted = [False] * len(iterables)
        while not all(exhausted):
            for i, it in enumerate(iterables_):
                try:
                    yield next(it)
                except StopIteration:
                    exhausted[i] = True
                    iterables_[i] = itertools.cycle(iterables[i])
                    # First elements may get repeated if one iterable is shorter than the others
                    yield next(iterables_[i])

    def __iter__(self):
        return self.repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)

class SampleRatioAwareDataLoader(object):
    '''
    This dataloader samples from multiple datasets according to specified ratios.
    If a dataset is exhausted before others, it resets (restarts) so the ratios are maintained.
    '''

    def __init__(self, dataloaders:dict, normalized_ratios:dict, merge_batches=False):
        
        self.dataloaders = dataloaders

        # Normalize the ratios
        self.normalized_ratios = normalized_ratios
        self.merge_batches = merge_batches
        print("TO Merge Batches: {}".format(self.merge_batches))

        # Determine the smallest dataset in terms of length
        self.smallest_dataset_key = min(self.dataloaders, key=lambda k: len(self.dataloaders[k]))
        self.smallest_dataset_len = len(self.dataloaders[self.smallest_dataset_key])
        min_ratio = self.normalized_ratios[self.smallest_dataset_key]
        ratios = {key: value / min_ratio for key, value in normalized_ratios.items()} # Make the ratio of smallest dataset 1

        # Calculating Dataloader size
        self.dataloader_size = int(sum([v * self.smallest_dataset_len for v in ratios.values()])) # dataloader size

        # Whole dataset will have this many samples
        print("Dataloader Presumed Normalized dataset will have total {} batches".format(self.dataloader_size))
        print("DEBUG: Normalized Ratios are: {}".format(normalized_ratios))

    def fetch_samples_from_datasets(self):
        """
        Fetch one batch of samples from each dataset and merge them into a single mixed batch
        """
        iterables_ = {key : iter(it) for key, it in self.dataloaders.items()}
        exhausted = {key: False for key in self.dataloaders.keys()}
        overlapping_keys = None
        running_count = 0

        while running_count < self.dataloader_size and not all(exhausted.values()):
            mixed_batch = {}
            overlapping_keys = None
            temp_batches = []
            
            for key, iterator in iterables_.items():
                try:
                    # Fetch one batch from the current dataset
                    retrieves = next(iterator)
                except StopIteration:
                    exhausted[key] = True
                    iterables_[key] = itertools.cycle(self.dataloaders[key])
                    retrieves = next(iterator)
                
                temp_batches.append(retrieves)
                
                # Determine if overlapping keys across datasets
                if overlapping_keys is None:
                    overlapping_keys = set(retrieves.keys())
                else:
                    overlapping_keys &= set(retrieves.keys())
            
            if overlapping_keys is None:
                raise Exception("Datasets have no overlapping keys")
            
            # Merge the batches into a single mixed batch
            for key in overlapping_keys:
                mixed_batch[key] = None
            
            for batch in temp_batches: # Concatenating them together
                for key in overlapping_keys:
                    if mixed_batch[key] is None:
                        mixed_batch[key] = batch[key]
                    else:
                        if isinstance(mixed_batch[key], torch.Tensor):
                            mixed_batch[key] = torch.cat((mixed_batch[key], batch[key]), dim=0)
                        else:
                            mixed_batch[key].extend(batch[key])      

            running_count += 1
            yield(mixed_batch)
        
    def ratio_aware_repetitive_roundrobin(self):
        """
        cycles through iterables but sample wise
        first yield first sample from first iterable then first sample from second iterable and so on
        then second sample from first iterable then second sample from second iterable and so on

        If one iterable is shorter than the others, it is repeated until all iterables are exhausted
        repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
        """
        # Repetitive roundrobin
        iterables_ = {key : iter(it) for key, it in self.dataloaders.items()}
        dataset_keys = list(self.dataloaders.keys())
        probabilities = [self.normalized_ratios[key] for key in dataset_keys]
        exhausted = {key: False for key in dataset_keys}
        running_count = 0 # To keep track of the number of samples yielded

        while running_count < self.dataloader_size and not all(exhausted.values()):
            # Pick a key at random
            chosen_key = random.choices(dataset_keys, weights=probabilities, k=1)[0]

            # print("DEBUG: Chosen key is: {}".format(chosen_key))
            try:
                yield next(iterables_[chosen_key])
            except StopIteration:
                exhausted[chosen_key] = True
                iterables_[chosen_key] = itertools.cycle(self.dataloaders[chosen_key])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[chosen_key])

            running_count += 1

    def __iter__(self):
        if self.merge_batches:
            return self.fetch_samples_from_datasets()
        else:
            return self.ratio_aware_repetitive_roundrobin()

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return self.dataloader_size

class MixedARTKITTINYU(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        
        sample_ratios = {
            'kitti': config.kitti_ratio,
            'art' : config.art_ratio,
            # 'nyu' : 1
        }

        # Dataset Configurations
        config = edict(config)
        self.config = config

        # Smallest Eats first: Making sure that there is at least batch size 1 for every dataset.
        # First, check if the batch size is at least the number of datasets. If not, raise error.
        self.batch_size = config.batch_size
        if len(sample_ratios) > self.batch_size:
            raise Exception("Error in MixedARTKITTINYU: Batch size cannot be less than the number of datasets")
        if config.workers < len(sample_ratios):
            raise Exception("Error in MixedARTKITTINYU: Number of workers cannot be less than the number of datasets")
        
        # Normalize the batch sizes across sample_ratios
        tot = sum(sample_ratios.values())
        self.normalized_ratios = {key: (values / tot) for key, values in sample_ratios.items()}
        self.normalized_batches = {key: value * self.batch_size for key, value in self.normalized_ratios.items()}
        self.normalized_workers = {key: value * config.workers  for key, value in self.normalized_ratios.items()}
        self.sorted_normalized_batches= list(sorted(self.normalized_batches.items(), key=lambda x:x[1]))
        self.sorted_normalized_workers= list(sorted(self.normalized_workers.items(), key=lambda x:x[1]))
        final_batches = {}
        final_workers = {}
        num_batches_left = self.batch_size
        num_workers_left = config.workers
        num_datasets = len(sample_ratios)

        # Assigning batches based on sorted normalized batches
        for i, (k, num_samples) in enumerate(self.sorted_normalized_batches): # num_samples here is a float
            if num_samples < 1:
                # Round up the batch size to one
                final_batches[k] = 1
                # Eat away from the next batch
                self.sorted_normalized_batches[i+1][1] -= (1 - num_samples) # Should always work because there is at least batch_size number of datasets
            else:
                if i < num_datasets -1 :
                    final_batches[k] = int(num_samples//1) # floor of num_samples
                else:
                    final_batches[k] = num_batches_left
            num_batches_left -= final_batches[k]
                
        # Assigning workers based on sorted normalized workers
        for i, (k, num_workers) in enumerate(self.sorted_normalized_workers): # num_workers here is a float
            if num_workers < 1:
                # Round up the worker count to one
                final_workers[k] = 1
                # Eat away from the next batch
                self.sorted_normalized_workers[i+1][1] -= (1 - num_workers)
            else:
                if i < num_datasets -1:
                    final_workers[k] = int(num_workers//1) # floor of num_workers
                else:
                    final_workers[k] = num_workers_left
            num_workers_left -= final_workers[k]
  
        # Converting the list back into a dictionary
        final_batches = dict(final_batches)
        final_workers = dict(final_workers)
        print(final_batches, final_workers)
        
        # Getting the config for each dataset
        conf_list = {}
        for k in sample_ratios.keys():
            conf_list[k] = change_dataset(edict(config), k) # This is not working for the ART Dataset. 
            conf_list[k].batch_size = final_batches[k] # Updating the adjusted batch size
            conf_list[k].workers = final_workers[k] # Updating the adjusted worker count
            print("Dataset {} will have a {} batch size, for {} workers".format(k, conf_list[k].batch_size, conf_list[k].workers))
        
        # Creating dataloaders
        if mode == 'train':
            dataloaders = {}
            for dataset_type, conf in conf_list.items():
                dataloaders[dataset_type] = DepthDataLoader(conf, mode, device=device).data
                print("Dataloader of {} has {} samples, {} batches".format(dataset_type, len(dataloaders[dataset_type])* dataloaders[dataset_type].batch_size, len(dataloaders[dataset_type])))
            # Ratio Aware Dataloader
            self.data = SampleRatioAwareDataLoader(dataloaders, self.normalized_ratios, self.config.merge_batches)
        else:
            # Testing Dataset
            art_test_conf = change_dataset(edict(config), 'art_test')
            art_test_conf.workers = art_test_conf.eval_workers
            # Make art_test default for testing
            self.data = DepthDataLoader(art_test_conf, mode, device=device).data

class MixedNYUKITTI(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        nyu_conf = change_dataset(edict(config), 'nyu')
        kitti_conf = change_dataset(edict(config), 'kitti')

        # make nyu default for testing
        self.config = config = nyu_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            nyu_loader = DepthDataLoader(
                nyu_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            kitti_loader = DepthDataLoader(
                kitti_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
        else:
            self.data = DepthDataLoader(nyu_conf, mode, device=device).data


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        """
        Initializes the dataset loader.

        Parameters:
        - config: Configuration object containing various settings (e.g., file paths, augmentation options).
        - mode: Specifies the mode ('train', 'online_eval', or other modes) that influences data loading.
        - transform: Optional transformation function to apply to samples.
        - is_for_online_eval: Boolean indicating if the dataset is being used for online evaluation.
        - kwargs: Additional arguments.
        """
        
        # Store configuration, mode, and transformation function
        self.config = config
        self.mode = mode
        self.transform = transform
        self.is_for_online_eval = is_for_online_eval

        # Load filenames based on mode (training or evaluation).
        # If mode is 'online_eval', load from `filenames_file_eval`, otherwise from `filenames_file`.
        if mode == 'eval':
            if config.dense_depth:
                eval_filename = config.filenames_file_eval_dense
            else:
                eval_filename = config.filenames_file_eval
            with open(eval_filename, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        # Initialize a tensor transformation method specific to the mode
        # self.to_tensor = ToTensor(mode)
        
        # Initialize the image reader object based on configuration
        if config.use_shared_dict:
            # Use a cached reader if `use_shared_dict` is enabled
            self.reader = CachedReader(config.shared_dict)
        else:
            # Default to a simple image reader
            self.reader = ImReader()

    def postprocess(self, sample):
        """
        Placeholder for any postprocessing that needs to be applied to each sample.
        By default, it just returns the sample as-is.
        """
        return sample

    def __getitem__(self, idx):
        """
        Retrieves a data sample at a specified index.

        Parameters:
        - idx: Index of the data sample to retrieve.

        Returns:
        - A dictionary containing the processed sample data.
        """
        
        # Get the path information for the sample
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])  # Parse the focal length from the path
        sample = {}  # Initialize an empty dictionary to store the sample data
        mask = False # Default if no mask exists
        has_valid_depth = False 
        has_valid_seg = False

        # Check if we are in training mode
        if self.mode == 'train':
            # Determine the image and depth paths based on dataset and configuration settings
            if self.config.dataset == 'kitti' and self.config.use_right and random.random() > 0.5:
                # For KITTI dataset with right camera images (50% probability)
                image_path = os.path.join(self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                seg_path = os.path.join(self.config.data_path, remove_leading_slash(sample_path.split()[3])[:-3] + 'npy')
                depth_path = os.path.join(self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            elif self.config.dataset[:3] == 'art':
                image_path = os.path.join(self.config.data_path, self.config.track)
                depth_path = os.path.join(self.config.gt_path, self.config.track)
                image_path = os.path.join(image_path, remove_leading_slash(sample_path.split()[0]))
                seg_path   = os.path.splitext(image_path)[0] + '.npy'
                depth_path = os.path.join(depth_path, remove_leading_slash(sample_path.split()[1]))
            else:
                # Standard case (left images)
                image_path = os.path.join(self.config.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(self.config.gt_path, remove_leading_slash(sample_path.split()[1]))
                seg_path = os.path.splitext(image_path)[0] + '.npy'

            # Load image and depth data
            image = self.reader.open(image_path)
            try:
                depth_gt = self.reader.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth_gt = False
                print("Depth file not found for image: ", depth_path)
                has_valid_depth = False
            w, h = image.size  # Get original dimensions of the image

            # Apply KITTI-specific cropping if enabled
            if self.config.do_kb_crop:
                height, width = image.height, image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                # Crop both image and depth ground truth
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # Random rotation augmentation
            if self.config.do_random_rotate and self.config.aug:
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            # Convert image and depth to float32 arrays and normalize
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # Scale depth values depending on the dataset
            if self.config.dataset == 'nyu':
                depth_gt /= 1000.0
            else:
                depth_gt /= 256.0

            # # Apply random crop and random translation if enabled in config
            # if self.config.aug and self.config.random_crop:
            #     image, depth_gt = self.random_crop(image, depth_gt, self.config.input_height, self.config.input_width)
            # if self.config.aug and self.config.random_translate:
            #     image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)

            # Load Seg data
            if self.config.use_segmentation:
                try:
                    seg_gt = np.load(seg_path)
                    # getting the index of the requested segmentation class 
                    segmentation_class_index = 13
                    # creating the segmentation mask for the specific label
                    seg_gt = (seg_gt == segmentation_class_index)

                    has_valid_seg = True
                except IOError:
                    seg_gt = False
                    print("Seg file not found for image: ", seg_path)
                    has_valid_seg = False
                
                # Additional preprocessing for training
                if has_valid_seg:
                    image, depth_gt, seg_gt = self.train_preprocess(image, depth_gt, seg_gt)
            else:
                # Additional preprocessing for training
                image, depth_gt = self.train_preprocess(image, depth_gt)

        else:
            # Loading for online evaluation or inference
            if self.config.dataset[:3] == 'art':
                data_path = self.config.data_path_eval if self.mode == 'eval' else self.config.data_path
                data_path = os.path.join(data_path, self.config.track)
            else:
                data_path = self.config.data_path_eval if self.mode == 'eval' else self.config.data_path
            image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
            seg_path   = os.path.join(data_path, remove_leading_slash(sample_path.split()[0])[:-3] + 'npy')
            image = np.asarray(self.reader.open(image_path), dtype=np.float32) / 255.0

            # For online evaluation, load depth data if available
            if self.mode == 'eval':
                if self.config.dataset[:3] == 'art':
                    gt_path = os.path.join(self.config.gt_path_eval, self.config.track)
                else:
                    gt_path = self.config.gt_path_eval
                depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
                try:
                    depth_gt = self.reader.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    has_valid_depth = False
                    # print("No depth available for {}".format(depth_path))
            
            # Process depth ground truth if valid
            if has_valid_depth:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt /= (1000.0 if self.config.dataset == 'nyu' else 256.0)

            # Load Seg data
            if self.config.use_segmentation:
                try:
                    seg_gt = np.load(seg_path)
                    # self.save_segmentation(np.squeeze(seg_gt), output_path = "/home/art/Depth-Anything/semseg/data/to_ashwin/test.png") #self.save_segmentation(seg_gt, "/home/art/Depth-Anything/semseg/data/to_ashwin/test.png")

                    # getting the index of the requested segmentation class 
                    segmentation_class_index = 13
                    # creating the segmentation mask for the specific label
                    seg_gt = (seg_gt == segmentation_class_index)

                    # print(seg_gt.sum())

                    has_valid_seg = True
                except IOError:
                    seg_gt = False
                    print("Seg file not found for image: ", seg_path)
                    has_valid_seg = False
            

        # Apply Art dataset-specific cropping.
        if has_valid_depth:
            if self.config.do_art_crop:

                # Image and depth map needs to be processed differently
                # Due to Dense Depth pre-cropped groundtruths

                # Process Image
                height, width, _ = image.shape
                crop_height = min(height, self.config.crop_remain)
                crop_width = min(width, self.config.art_width)
                bottom_margin = (height - crop_height) // 2
                top_margin = height - bottom_margin
                left_margin = (width - crop_width) // 2
                right_margin = width - left_margin

                # Crop Image
                image = image[bottom_margin:top_margin, left_margin:right_margin, ...]

                if height < self.config.crop_remain or width < self.config.art_width:
                    # For image and gt depth, resize. Image using bilinear, depth using nearest neighbor
                    image = cv2.resize(image, (self.config.art_width, self.config.crop_remain), interpolation=cv2.INTER_LINEAR)

                # Process groundtruth
                height, width, _ = depth_gt.shape
                crop_height = min(height, self.config.crop_remain)
                crop_width = min(width, self.config.art_width)
                bottom_margin = (height - crop_height) // 2
                top_margin = height - bottom_margin
                left_margin = (width - crop_width) // 2
                right_margin = width - left_margin

                # Crop both seg masks and depth ground truth
                if has_valid_seg:
                    seg_gt   = self.pad_or_crop_image(seg_gt, crop_height, crop_width)

                depth_gt = depth_gt[bottom_margin:top_margin, left_margin:right_margin, ...] 

                if height < self.config.crop_remain or width < self.config.art_width:
                    # For image and gt depth, resize. Image using bilinear, depth using nearest neighbor
                    depth_gt = cv2.resize(depth_gt, (self.config.art_width, self.config.crop_remain), interpolation=cv2.INTER_NEAREST)
                    if has_valid_seg:
                        seg_gt   = cv2.resize(seg_gt  , (self.config.art_width, self.config.crop_remain), interpolation=cv2.INTER_NEAREST)

            # Performing Shifting on Art Dataset
            # if self.config.dataset[:3] == 'art':
            if self.config.dataset[:5] == 'kitti' and self.config.get("kitti_scale_factor", False):
                # depth_gt = depth_gt - self.config.art_pred_shift # Will be added back on for eval
                depth_gt = depth_gt * self.config.kitti_scale_factor * float(401.5879 / focal) # Will be added back on for eval
                # The below statement will take care of negative values.
            mask = np.logical_and(depth_gt > self.config.min_depth, depth_gt < self.config.max_depth).squeeze()[None, ...]
        
        if self.mode == 'train':
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'seg_mask':np.expand_dims(seg_gt, 0), **sample} # NOT SURE HOW TRAINER HANDLES INVALID SAMPLES
        else:
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, 'mask': mask, 'seg_mask':np.expand_dims(seg_gt, 0)}
            if not sample['has_valid_depth'] or isinstance(sample["mask"], int):
                return {'image': False, 'depth': False, 'focal': False, 'has_validd_epth': False, 'mask': False, 'seg_mask':False}
                
        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
        if self.transform:
            sample = self.transform(sample)
        return sample

# Assume colors are defined using a colormap for class values 0-19
    def save_segmentation(self, seg_gt, output_path = "/home/art/Depth-Anything/semseg/data/to_ashwin/test.png"):
        # Define a colormap for the classes (0-19)
        cmap = plt.colormaps['tab20']  # Use matplotlib's tab20 colormap
        colors = [cmap(i)[:3] for i in range(20)]  # Extract RGB colors (ignoring alpha)
        colors = (np.array(colors) * 255).astype(np.uint8)  # Convert to 0-255 range

        # Create an RGB image from the segmentation mask
        seg_color = np.zeros((seg_gt.shape[0], seg_gt.shape[1], 3), dtype=np.uint8)
        for class_value in range(20):  # For classes 0-19
            seg_color[seg_gt == class_value] = colors[class_value]

        # Save the segmentation image to the output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        cv2.imwrite(output_path, cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))  # Save as BGR for OpenCV compatibility
        print(f"Segmentation saved to {output_path}")
        raise NotImplementedError
    def pad_or_crop_image(self, image, target_height, target_width):
        """Pad or crop an image to the desired size (target_height, target_width)."""
        image = np.squeeze(image)
        if image.dtype == bool:
            image = image.astype(np.uint8)
        h, w = image.shape

        # If the image is smaller, pad with zeros
        if h < target_height or w < target_width:
            top_pad = max(0, (target_height - h) // 2)
            bottom_pad = max(0, target_height - h - top_pad)
            left_pad = max(0, (target_width - w) // 2)
            right_pad = max(0, target_width - w - left_pad)
            image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # If the image is larger, crop to the target size
        if h > target_height or w > target_width:
            top_crop = max(0, (h - target_height) // 2)
            bottom_crop = top_crop + target_height
            left_crop = max(0, (w - target_width) // 2)
            right_crop = left_crop + target_width
            image = image[top_crop:bottom_crop, left_crop:right_crop]
            
        return image

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt, seg_gt=None):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()
                if seg_gt is not None:
                    seg_gt = (seg_gt[:, :, ::-1]).copy()

            # # Random gamma, brightness, color augmentation
            # do_augment = random.random()
            # if do_augment > 0.5:
            #     image = self.augment_image(image)
        if seg_gt is not None:
            return image, depth_gt, seg_gt
        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, config=None,do_normalize=False, size=None):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        self.preparefornet = None
        if size is not None:
            self.resize = transforms.Resize(size=size, interpolation=Image.BILINEAR)
        else:
            self.resize = nn.Identity()
            
        # Add in Image Augmentations here
        if mode == 'train' and config and config["apply_augmentations"]:
            self.augment_probability = config.get("augmentation_probability", 0.5)
            
            # List of Augmentations to Perform
            self.augmentations = [
                transforms.ColorJitter(
                    brightness=config.get("brightness", 0.2),
                    contrast=config.get("contrast", 0.2),
                    saturation=config.get("saturation", 0.1),
                    hue=config.get("hue", 0.2)
                ),
                # v2.RandomInvert(p =1.0),
                v2.RandomEqualize(p=1.0),
                # v2.RandomSolarize(threshold=192.0/255.0, p=1.0)
            ]
            
            # augmentation_transforms = []
            # if config and config.apply_jitter:
            #     probability = config.get("probability", 0.5)
            #     augmentation_transforms.append(transforms.RandomApply(
            #         [transforms.ColorJitter(
            #             brightness=config.get("brightness", 0.15),
            #             contrast=config.get("contrast", 0.5),
            #             saturation=config.get("saturation", 0.5),
            #             hue=config.get("hue", 0.1)
            #         )],
            #         p=probability
            #     ))
            
            # if config and config.apply_invert:
            #     augmentation_transforms.append(
            #         v2.RandomInvert(p =probability)
            #     )

            # # equalizer = v2.RandomEqualize()
            # # solarizer = v2.RandomSolarize(threshold=192.0)
            # self.augment_transform = transforms.Compose(augmentation_transforms)
        else:
            self.augmentations = []
            self.augment_probability = 0
            # self.augment_transform = nn.Identity()
        
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        
        if self.mode == 'train' and self.augmentations and random.random() < self.augment_probability:
            augmentation = random.choice(self.augmentations)
            image = augmentation(image)
        image = self.normalize(image)
        image = self.resize(image)
        
        if self.preparefornet is not None:
            image = self.preparefornet(image)
        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            
            depth = self.to_tensor(depth)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            # print("DEBUG: ", sample.keys(), " MODE: ", self.mode)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal}

            # return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
            #         'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
