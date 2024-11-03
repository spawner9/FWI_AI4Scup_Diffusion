# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glob
import sys 
from PIL import Image
from skimage.transform import resize


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset_pytorch(config):
  """
  Create data loaders for training and evaluation. But using the torch.Dataset
  """
  # Compute batch size for this worker.
  if config.data.dataset == 'Marmousi':
    data_dir = "/home/caoxiang/Desktop/Datasets/Marmousi/"
    train_split_name = 'train'
    eval_split_name = 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
          super(Create_Custom_Dataset, self).__init__()
          self.image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
          self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图片大小
            transforms.ToTensor(),  # 将图片转换为PyTorch张量
        ])
          
      def __len__(self):
          return len(self.image_files)

      def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('L')  # 'L' mode means single-channel (grayscale)
        image = self.transform(image)
        
        return image

    train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)

  
  elif config.data.dataset == 'KIT4':
    data_dir = "/home/caoxiang/Desktop/Datasets/KIT4/samples_fig/"
    train_split_name = 'train'
    eval_split_name = 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
          super(Create_Custom_Dataset, self).__init__()
          self.image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
          self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图片大小
            transforms.ToTensor(),  # 将图片转换为PyTorch张量
        ])
          
      def __len__(self):
          return len(self.image_files)

      def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('L')  # 'L' mode means single-channel (grayscale)
        image = self.transform(image)
        
        return image

    train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)
  
  elif config.data.dataset == 'AI4Scup2_full':
    data_dir = "/home/caoxiang/Desktop/Datasets/AI4Scup2_full/speed"
    train_split_name = 'train' 

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
          super(Create_Custom_Dataset, self).__init__()
          self.image_files = glob.glob(os.path.join(data_dir, split, '*.npy'))
          self.speed_max, self.speed_min = 1595.1279, 1408.692

      def __len__(self):
          return len(self.image_files)

      def __getitem__(self, idx):
        image_full = np.load(self.image_files[idx])
        image = resize(image_full[90:390, 90:390], (256, 256), mode='reflect', anti_aliasing=True)
        image = 2 * (image - self.speed_min) / (self.speed_max - self.speed_min) -1 

        return torch.tensor(image).unsqueeze(0)
        
    train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_loader = data.DataLoader(train_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)

  elif config.data.dataset == 'AI4Scup2':
    data_dir = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed"
    train_split_name = 'train'
    eval_split_name = 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
          super(Create_Custom_Dataset, self).__init__()
          self.image_files = glob.glob(os.path.join(data_dir, split, '*.npy'))
          self.speed_max, self.speed_min = 1610, 1400
          self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将 NumPy 数组转换为 PyTorch 张量
        ])

      def __len__(self):
          return len(self.image_files)

      def __getitem__(self, idx):
        image_full = np.load(self.image_files[idx])
        image = resize(image_full[90:390, 90:390], (256, 256), mode='reflect', anti_aliasing=True)
        image = (image - self.speed_min) / (self.speed_max - self.speed_min)
        image = np.clip(image, 0, 1)
        image = self.transform(image)
        
        return image

    train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)
    
  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  
  
  
  return train_loader, eval_loader, data_dir 



def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
      
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder
