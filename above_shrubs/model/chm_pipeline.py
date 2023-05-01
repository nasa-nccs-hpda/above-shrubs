import os
import re
import sys
import time
import logging
import argparse
import omegaconf
import rasterio
import numpy as np
import xarray as xr
import rioxarray as rxr

from glob import glob
from pathlib import Path

from tensorflow_caney.utils.system import set_gpu_strategy, set_mixed_precision, set_xla
from tensorflow_caney.utils.data import get_dataset_filenames, get_mean_std_dataset
from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.model import load_model, get_model
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.inference import regression_inference

from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from above_shrubs.model.regression_dataloader import RegressionDataLoaderSRLite

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = np

__status__ = "Development"

class CHMPipeline(CNNRegression):

    # -------------------------------------------------------------------------
    # preprocess
    # -------------------------------------------------------------------------
    def preprocess(self):
        """
        Perform general preprocessing.
        """
        logging.info('Starting preprocessing stage')

        # Calculate mean and std values for training
        data_filenames = get_dataset_filenames(self.images_dir)
        label_filenames = get_dataset_filenames(self.labels_dir)
        logging.info(f'Mean and std values from {len(data_filenames)} files.')

        # Temporarily disable standardization and augmentation
        current_standardization = self.conf.standardization
        self.conf.standardization = None
        metadata_output_filename = os.path.join(
            self.model_dir, f'mean-std-{self.conf.experiment_name}.csv')
        os.makedirs(self.model_dir, exist_ok=True)

        # Set main data loader
        main_data_loader = RegressionDataLoaderSRLite(
            data_filenames, label_filenames, self.conf, False
        )

        # Get mean and std array
        mean, std = get_mean_std_dataset(
            main_data_loader.train_dataset, metadata_output_filename)
        logging.info(f'Mean: {mean.numpy()}, Std: {std.numpy()}')

        # Re-enable standardization for next pipeline step
        self.conf.standardization = current_standardization

        logging.info('Done with preprocessing stage')

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    def train(self) -> None:

        self.logger.info('Starting training stage')

        # Set hardware acceleration options
        gpu_strategy = set_gpu_strategy(self.conf.gpu_devices)
        set_mixed_precision(self.conf.mixed_precision)
        set_xla(self.conf.xla)

        # Get data and label filenames for training
        data_filenames = get_dataset_filenames(self.images_dir)
        label_filenames = get_dataset_filenames(self.labels_dir)
        assert len(data_filenames) == len(label_filenames), \
            'Number of data and label filenames do not match'

        logging.info(
            f'Data: {len(data_filenames)}, Label: {len(label_filenames)}')

        # Set main data loader
        main_data_loader = RegressionDataLoaderSRLite(
            data_filenames, label_filenames, self.conf
        )

        # Set multi-GPU training strategy
        with gpu_strategy.scope():

            if self.conf.transfer_learning == 'feature-extraction':

                # get full model for training
                model = get_model(self.conf.model)
                model.trainable = False
                pretrained = load_model(
                    model_filename=self.conf.transfer_learning_weights,
                    model_dir=self.model_dir
                )
                model.set_weights(pretrained.get_weights())
                logging.info(
                    f"Load weights from {self.conf.transfer_learning_weights}")

                model.trainable = True
                model.compile(
                    loss=get_loss(self.conf.loss),
                    optimizer=get_optimizer(
                        self.conf.optimizer)(self.conf.learning_rate),
                    metrics=get_metrics(self.conf.metrics)
                )

            elif self.conf.transfer_learning == 'fine-tuning':

                # get full model for training
                model = get_model(self.conf.model)
                model.trainable = False
                pretrained = load_model(
                    model_filename=self.conf.transfer_learning_weights,
                    model_dir=self.model_dir
                )
                model.set_weights(pretrained.get_weights())
                logging.info(
                    f"Load weights from {self.conf.transfer_learning_weights}")

                # Freeze all the layers before the `fine_tune_at` layer
                for layer in model.layers[
                        :self.conf.transfer_learning_fine_tune_at]:
                    layer.trainable = False

                model.compile(
                    loss=get_loss(self.conf.loss),
                    optimizer=get_optimizer(
                        self.conf.optimizer)(self.conf.learning_rate),
                    metrics=get_metrics(self.conf.metrics)
                )

            else:
                # Get and compile the model
                model = get_model(self.conf.model)
                model.compile(
                    loss=get_loss(self.conf.loss),
                    optimizer=get_optimizer(
                        self.conf.optimizer)(self.conf.learning_rate),
                    metrics=get_metrics(self.conf.metrics)
                )

        model.summary()

        # Fit the model and start training
        model.fit(
            main_data_loader.train_dataset,
            validation_data=main_data_loader.val_dataset,
            epochs=self.conf.max_epochs,
            steps_per_epoch=main_data_loader.train_steps,
            validation_steps=main_data_loader.val_steps,
            callbacks=get_callbacks(self.conf.callbacks)
        )
        logging.info(f'Done with training, models saved: {self.model_dir}')

        return
