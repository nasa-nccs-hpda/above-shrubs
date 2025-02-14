import os
import re
import sys
import logging
import numpy as np
import rioxarray as rxr
import tensorflow as tf
from typing import Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow_caney.utils.data import get_mean_std_metadata, \
    standardize_image, normalize_image, normalize_meanstd
from tensorflow_caney.utils.augmentations import center_crop


AUTOTUNE = tf.data.experimental.AUTOTUNE

__all__ = ["RegressionDataLoader"]


class RegressionDataLoader(object):

    def __init__(
                self,
                conf,
                train_data_filenames: list,
                train_label_filenames: list,
                test_data_filenames: list,
                test_label_filenames: list,
                train_step: bool = True,
            ):

        # Set configuration variables
        self.conf = conf
        self.train_step = train_step

        # Filename with mean and std metadata
        self.metadata_output_filename = os.path.join(
            self.conf.model_dir, f'mean-std-{self.conf.experiment_name}.csv')
        self.mean = None
        self.std = None

        # Set data filenames
        self.train_x = train_data_filenames
        self.train_y = train_label_filenames

        # Set test filenames
        self.test_x = test_data_filenames
        self.test_y = test_label_filenames

        # Disable AutoShard, data lives in memory, use in memory options
        self.options = tf.data.Options()
        self.options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF

        # Total size of the dataset
        total_train_size = len(train_data_filenames)
        logging.info(f'Training Dataset Size: {total_train_size}')

        # Checking some parameters
        logging.info(
            f'Crop {self.conf.center_crop} Augment {self.conf.augment}')

        # If this is not a training step (e.g preprocess, predict)
        if not train_step:

            # Initialize training dataset
            self.train_dataset = self.tf_dataset(
                self.train_x, self.train_y,
                read_func=self.tf_data_loader,
                repeat=False, batch_size=self.conf.batch_size
            )
            self.train_dataset = self.train_dataset.with_options(self.options)

        # Else, if this is a training step (e.g. train)
        else:

            # Total size of the dataset
            total_test_size = len(test_data_filenames)
            logging.info(f'Test Dataset Size: {total_test_size}')

            # Calculate training steps
            self.train_steps = len(self.train_x) // self.conf.batch_size
            self.test_steps = len(self.test_x) // self.conf.batch_size

            if len(self.train_x) % self.conf.batch_size != 0:
                self.train_steps += 1
            if len(self.test_x) % self.conf.batch_size != 0:
                self.test_steps += 1

            # Initialize training dataset
            self.train_dataset = self.tf_dataset(
                self.train_x, self.train_y,
                read_func=self.tf_data_loader,
                repeat=True, batch_size=self.conf.batch_size
            )
            self.train_dataset = self.train_dataset.with_options(self.options)

            # Initialize validation dataset
            self.test_dataset = self.tf_dataset(
                self.test_x, self.test_y,
                read_func=self.tf_data_loader,
                repeat=True, batch_size=self.conf.batch_size
            )
            self.test_dataset = self.test_dataset.with_options(self.options)

        # Load mean and std metrics, only if training and fixed standardization
        if train_step and self.conf.standardization in ['global', 'mixed']:
            self.mean, self.std = get_mean_std_metadata(
                self.metadata_output_filename)

    def tf_dataset(
                self,
                x: list,
                y: list,
                read_func: Any,
                repeat=True,
                batch_size=64
            ) -> Any:
        """
        Fetch tensorflow dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(2048)
        dataset = dataset.map(read_func, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        if repeat:
            dataset = dataset.repeat()
        return dataset

    def tf_data_loader(self, x, y):
        """
        Initialize TensorFlow dataloader.
        """
        def _loader(x, y):
            x, y = self.load_data(x.decode(), y.decode())
            return x.astype(np.float32), y.astype(np.float32)
        x, y = tf.numpy_function(_loader, [x, y], [tf.float32, tf.float32])
        x.set_shape([
            self.conf.tile_size,
            self.conf.tile_size,
            len(self.conf.output_bands)]
        )
        y.set_shape([
            self.conf.tile_size,
            self.conf.tile_size,
            self.conf.n_classes]
        )
        return x, y

    def load_data(self, x, y):
        """
        Load data on training loop.
        """
        extension = Path(x).suffix

        if self.conf.metadata_regex is not None:
            year_match = re.search(r'(\d{4})(\d{2})(\d{2})', x)
            timestamp = str(int(year_match.group(2)))

        # Read data
        if extension == '.npy':
            # TODO: make channel dim more dynamic
            # if 0 < 1 then channel last, etc.
            x = np.load(x)
            y = np.load(y)
            # print(y.min(), y.max())
        elif extension == '.tif':
            x = np.moveaxis(rxr.open_rasterio(x).data, 0, -1)
            y = np.moveaxis(rxr.open_rasterio(y).data, 0, -1)
        else:
            sys.exit(f'{extension} format not supported.')

        if len(y.shape) < 3:
            y = np.expand_dims(y, axis=-1)

        # Normalize labels, default is diving by 1.0
        x = normalize_image(x, self.conf.normalize)
        y = normalize_image(y, self.conf.normalize_label)

        # Standardize
        if self.conf.metadata_regex is not None:
            x = normalize_meanstd(
                x, self.metadata[timestamp], subtract='median'
            )
        elif self.conf.standardization is not None:
            x = standardize_image(
                x, self.conf.standardization, self.mean, self.std)

        # Crop
        if self.conf.center_crop:
            x = center_crop(x, (self.conf.tile_size, self.conf.tile_size))
            y = center_crop(y, (self.conf.tile_size, self.conf.tile_size))

        # Augment
        if self.conf.augment:

            if np.random.random_sample() > 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)
            if np.random.random_sample() > 0.5:
                x = np.flipud(x)
                y = np.flipud(y)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 1)
                y = np.rot90(y, 1)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 2)
                y = np.rot90(y, 2)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 3)
                y = np.rot90(y, 3)

        return x, y


class RegressionDataLoaderSRLite(RegressionDataLoader):

    # we modify the load_data function for this use case
    def load_data(self, x, y):
        """
        Load data on training loop.
        """

        # Read data
        x = np.load(x)
        y = np.load(y)

        # channels-first conversion
        if x.shape[0] < x.shape[-1]:
            x = np.moveaxis(x, 0, -1)

        if y.shape[0] < y.shape[-1]:
            y = np.moveaxis(y, 0, -1)

        # try this experiment
        # small relative value to height of pixels with 0
        # small random epsilon
        y[y < 0] = 0
        y = abs(y)

        # Normalize labels, default is diving by 1.0
        x = normalize_image(x, self.conf.normalize)
        y = normalize_image(y, self.conf.normalize_label)

        # Simple standardization, replace based on your own project
        if self.conf.standardization is not None:
            x = standardize_image(
                x, self.conf.standardization, self.mean, self.std)

        # Augment
        if self.conf.augment:

            if np.random.random_sample() > 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)
            if np.random.random_sample() > 0.5:
                x = np.flipud(x)
                y = np.flipud(y)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 1)
                y = np.rot90(y, 1)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 2)
                y = np.rot90(y, 2)
            if np.random.random_sample() > 0.5:
                x = np.rot90(x, 3)
                y = np.rot90(y, 3)

        return x, y
