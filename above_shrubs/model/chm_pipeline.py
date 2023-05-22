import os
import logging
import numpy as np
import rioxarray as rxr
from pathlib import Path
from tqdm import tqdm

from tensorflow_caney.utils.system import set_gpu_strategy, \
    set_mixed_precision, set_xla, seed_everything
from tensorflow_caney.utils.data import get_dataset_filenames, \
    get_mean_std_dataset
from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.model import load_model, get_model
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.utils.data import get_mean_std_metadata, \
    standardize_image

from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from above_shrubs.model.regression_dataloader import RegressionDataLoaderSRLite
from above_shrubs.model.config import CHMConfig as Config

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = np

__status__ = "Development"


class CHMPipeline(CNNRegression):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config_filename, logger=None):

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        # Configuration file intialization
        self.conf = self._read_config(config_filename, Config)

        # output directory to store metadata and artifacts
        self.metadata_dir = os.path.join(self.conf.data_dir, 'metadata')
        self.logger.info(f'Metadata dir: {self.metadata_dir}')

        # Set output directories and locations
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        self.logger.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        self.logger.info(f'Model dir: {self.labels_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.metadata_dir, self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # _tif_to_numpy
    # -------------------------------------------------------------------------
    def _tif_to_numpy(self, data_filenames, label_filenames, output_dir):
        """
        Convert TIF to NP.
        """
        for data_filename, label_filename in tqdm(
                                    zip(data_filenames, label_filenames)):

            # open the imagery
            image = rxr.open_rasterio(data_filename).values
            label = rxr.open_rasterio(label_filename).values

            if np.isnan(label).any():
                continue

            # get output filenames
            image_output_dir = os.path.join(output_dir, 'images')
            os.makedirs(image_output_dir, exist_ok=True)

            label_output_dir = os.path.join(output_dir, 'labels')
            os.makedirs(label_output_dir, exist_ok=True)

            # save the new arrays
            np.save(
                os.path.join(
                    image_output_dir,
                    f'{Path(data_filename).stem}.npy'
                ), image)
            np.save(
                os.path.join(
                    label_output_dir,
                    f'{Path(label_filename).stem}.npy'
                ), label)
        return

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(self):
        """
        Convert .tif into numpy files.
        TODO: Make it run in parallel, a tile per core.
        """
        # Get data and label filenames for training
        if self.conf.train_data_dir is not None:
            train_data_filenames = get_dataset_filenames(
                self.conf.train_data_dir, ext='*.tif')
            train_label_filenames = get_dataset_filenames(
                self.conf.train_label_dir, ext='*.tif')
            assert len(train_data_filenames) == len(train_label_filenames), \
                'Number of data and label filenames do not match'
            print(f'Converting {len(train_data_filenames)} tiles')
            self._tif_to_numpy(
                train_data_filenames,
                train_label_filenames,
                self.conf.data_dir
            )

        # Get data and label filenames for training
        if self.conf.test_data_dir is not None:
            test_data_filenames = get_dataset_filenames(
                self.conf.test_data_dir, ext='*.tif')
            test_label_filenames = get_dataset_filenames(
                self.conf.test_label_dir, ext='*.tif')
            assert len(test_data_filenames) == len(test_label_filenames), \
                'Number of data and label filenames do not match'
            print(f'Converting {len(test_data_filenames)} tiles')
            self._tif_to_numpy(
                test_data_filenames,
                test_label_filenames,
                self.conf.test_dir
            )
        return

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

    # -------------------------------------------------------------------------
    # validate
    # -------------------------------------------------------------------------
    def validate(self) -> None:
        """
        Perform prediction - tiles and tif images
        """
        # Load model for inference
        model = load_model(
            model_filename=self.conf.model_filename,
            model_dir=self.conf.model_dir
        )
        logging.info(f'Loaded model from {self.conf.model_dir}')

        # create output directory
        os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # Retrieve mean and std, there should be a more ideal place
        if self.conf.standardization in ["global", "mixed"]:
            mean, std = get_mean_std_metadata(
                os.path.join(
                    self.model_dir,
                    f'mean-std-{self.conf.experiment_name}.csv'
                )
            )
            logging.info(f'Mean: {mean}, Std: {std}')
        else:
            mean = None
            std = None

        # get filenames for prediction
        test_data_filenames = get_dataset_filenames(
            os.path.join(self.conf.test_dir, 'images'), ext='*.npy')
        test_label_filenames = get_dataset_filenames(
            os.path.join(self.conf.test_dir, 'labels'), ext='*.npy')
        logging.info(f'Loaded {len(test_data_filenames)} tiles')

        # create array with data to predict
        test_data = []
        test_labels = []
        for data_filename, label_filename in \
                zip(test_data_filenames, test_label_filenames):

            # read data
            test_data_tile = np.moveaxis(
                np.load(data_filename), 0, -1)
            if self.conf.standardization is not None:
                test_data_tile = standardize_image(
                    test_data_tile, self.conf.standardization,
                    mean, std
                )

            # read label
            test_data_label = np.load(label_filename)

            test_data.append(test_data_tile)
            test_labels.append(test_data_label)

        # create array and reshape it
        test_data = np.array(test_data)
        test_labels = np.moveaxis(np.array(test_labels), 1, -1)
        logging.info(f'Reshaped data: {test_data.shape}, {test_labels.shape}')

        # perform prediction
        predictions = model.predict(
            test_data, batch_size=self.conf.pred_batch_size)
        logging.info(f'Shape after prediction {predictions.shape}')

        results = model.evaluate(
            test_data, test_labels, batch_size=self.conf.pred_batch_size)
        logging.info(f'Results output: {results}')

        predictions = np.squeeze(predictions)
        test_labels = np.squeeze(test_labels)
        logging.info(f'Shape of predictions {predictions.shape}')

        for index, filename in tqdm(enumerate(test_data_filenames)):

            output_filename = os.path.join(
                self.conf.inference_save_dir,
                f'{Path(filename).stem}.npy'
            )
            np.save(output_filename, predictions[index, :, :])

        return
