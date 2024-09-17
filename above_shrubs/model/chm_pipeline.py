import os
import re
import time
import logging
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import rioxarray as rxr

from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from tensorflow.data import Dataset
from pygeotools.lib import iolib, warplib
from multiprocessing import Pool, cpu_count
from omegaconf.listconfig import ListConfig

from sklearn.metrics import r2_score, mean_absolute_error, \
    mean_squared_error, mean_absolute_percentage_error

from tensorflow_caney.utils.system import set_gpu_strategy, \
    set_mixed_precision, set_xla, seed_everything
from tensorflow_caney.utils.data import modify_bands, \
    get_dataset_filenames, get_mean_std_dataset, \
    get_mean_std_metadata, read_metadata, standardize_image
from tensorflow_caney.utils import indices
from tensorflow_caney.utils.losses import get_loss
from tensorflow_caney.utils.model import load_model, get_model
from tensorflow_caney.utils.optimizers import get_optimizer
from tensorflow_caney.utils.metrics import get_metrics
from tensorflow_caney.utils.callbacks import get_callbacks
from tensorflow_caney.model.pipelines.cnn_regression import CNNRegression
from tensorflow_caney.inference import regression_inference

from above_shrubs.model.regression_dataloader import RegressionDataLoaderSRLite
from above_shrubs.model.config import CHMConfig as Config

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
xp = np

__status__ = "Development"


class CHMPipeline(CNNRegression):
    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                config_filename: str,
                model_filename: str = None,
                output_dir: str = None,
                inference_regex_list: list = None,
                default_config: str = 'templates/chm_cnn_default.yaml',
                logger=None
            ):
        """Constructor method
        """

        # Set logger
        self.logger = logger if logger is not None else self._set_logger()

        logging.info('Initializing CHMPipeline')

        # Configuration file intialization
        if config_filename is None:
            config_filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                default_config)
            logging.info(f'Loading default config: {config_filename}')

        self.conf = self._read_config(config_filename, Config)

        # rewrite model filename option if given from CLI
        if model_filename is not None:
            assert os.path.exists(model_filename), \
                f'{model_filename} does not exist.'
            self.conf.model_filename = model_filename

        # rewrite output directory if given from CLI
        if output_dir is not None:
            self.conf.inference_save_dir = output_dir
            os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # rewrite inference regex list
        if inference_regex_list is not None:
            self.conf.inference_regex_list = inference_regex_list

        # output directory to store metadata and artifacts
        self.metadata_dir = os.path.join(self.conf.data_dir, 'metadata')
        self.logger.info(f'Metadata dir: {self.metadata_dir}')

        # Set output directories and locations
        self.images_dir = os.path.join(self.conf.data_dir, 'images')
        self.logger.info(f'Images dir: {self.images_dir}')

        self.labels_dir = os.path.join(self.conf.data_dir, 'labels')
        self.logger.info(f'Labels dir: {self.labels_dir}')

        self.model_dir = self.conf.model_dir
        self.logger.info(f'Model dir: {self.model_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.metadata_dir, self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        logging.info(f'Output dir: {self.conf.inference_save_dir}')

        # save configuration into the model directory
        try:
            OmegaConf.save(
                self.conf, os.path.join(self.model_dir, 'config.yaml'))
        except PermissionError:
            logging.info('No permissions to save config, skipping step.')

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # add_dtm
    # -------------------------------------------------------------------------
    def add_dtm(self, raster_filename: str, raster, dtm_filename: str):

        # warp dtm to match raster
        warp_ds_list = warplib.memwarp_multi_fn(
            [raster_filename, dtm_filename], res=raster_filename,
            extent=raster_filename, t_srs=raster_filename, r='average',
            dst_ndv=int(raster.rio.nodata), verbose=False
        )
        dtm_ma = iolib.ds_getma(warp_ds_list[1])

        # Drop image band to allow for a merge of mask
        dtm = raster.drop(
            dim="band",
            labels=raster.coords["band"].values[1:],
        )
        dtm.coords['band'] = [raster.shape[0] + 1]

        # Get metadata to save raster
        dtm = xr.DataArray(
            np.expand_dims(dtm_ma, axis=0),
            name='dtm',
            coords=dtm.coords,
            dims=dtm.dims,
            attrs=dtm.attrs
        ).fillna(raster.rio.nodata)
        dtm = dtm.where(raster[0, :, :] > 0, int(raster.rio.nodata))

        # concatenate the bands together
        dtm = xr.concat([raster, dtm], dim="band")
        dtm = dtm.where(dtm > 0, int(raster.rio.nodata))

        # additional clean-up for the imagery
        dtm.where(
            dtm.any(dim='band') != True,  # noqa: E712
            int(raster.rio.nodata)
        )
        dtm = dtm.where(dtm > 0, int(raster.rio.nodata))
        dtm.attrs['long_name'] = dtm.attrs['long_name'] + ("DTM",)
        return dtm

    def add_cloudmask(
                self,
                raster,
                cloudmask_filename: str,
                nodata_value: int = -9999
            ):
        # TODO: try: except, I need to know the error
        cloud_raster = rxr.open_rasterio(cloudmask_filename)
        raster = raster.where(cloud_raster[0, :, :] == 0, nodata_value)
        return raster

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_filenames(
                self,
                data_regex: str,
                allow_empty: bool = False
            ) -> list:
        """
        Get filename from list of regexes
        """
        # get the paths/filenames of the regex
        filenames = []
        if isinstance(data_regex, list) or isinstance(data_regex, ListConfig):
            for regex in data_regex:
                if regex[-3:] == 'csv':
                    filenames.extend(pd.read_csv(regex).iloc[:, 0].tolist())
                else:
                    filenames.extend(glob(regex))
        else:
            filenames = glob(data_regex)

        # in some cases, we need to assert if we found files
        # in other cases, we just ignore the fact that we did not find any
        if not allow_empty:
            assert len(filenames) > 0, f'No files under {data_regex}'
        return sorted(filenames)

    # -------------------------------------------------------------------------
    # _tif_to_numpy
    # -------------------------------------------------------------------------
    def _tif_to_numpy(self, data_filename, label_filename, output_dir):
        """
        Convert TIF to NP.
        """
        # open the imagery
        image = rxr.open_rasterio(data_filename).values
        label = rxr.open_rasterio(label_filename).values

        if np.isnan(label).any():
            return

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
    # _xbatch
    # -------------------------------------------------------------------------
    def _xbatch(self, iterable, batch_size=1):
        iter = len(iterable)
        for ndx in range(0, iter, batch_size):
            yield iterable[ndx:min(ndx + batch_size, iter)]

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(self):
        """
        Convert .tif into numpy files.
        """
        # Get data and label filenames for training
        if self.conf.train_data_dir is not None:
            p = Pool(processes=cpu_count())
            train_data_filenames = get_dataset_filenames(
                self.conf.train_data_dir, ext='*.tif')
            train_label_filenames = get_dataset_filenames(
                self.conf.train_label_dir, ext='*.tif')
            assert len(train_data_filenames) == len(train_label_filenames), \
                'Number of data and label filenames do not match'
            logging.info(f'{len(train_data_filenames)} files from TIF to NPY')
            p.starmap(
                self._tif_to_numpy,
                zip(
                    train_data_filenames,
                    train_label_filenames,
                    repeat(self.conf.data_dir)
                )
            )

        # Get data and label filenames for training
        if self.conf.test_data_dir is not None:
            p = Pool(processes=cpu_count())
            test_data_filenames = get_dataset_filenames(
                self.conf.test_data_dir, ext='*.tif')
            test_label_filenames = get_dataset_filenames(
                self.conf.test_label_dir, ext='*.tif')
            assert len(test_data_filenames) == len(test_label_filenames), \
                'Number of data and label filenames do not match'
            logging.info(f'{len(test_data_filenames)} files from TIF to NPY')
            p.starmap(
                self._tif_to_numpy,
                zip(
                    test_data_filenames,
                    test_label_filenames,
                    repeat(self.conf.test_dir)
                )
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
    # predict
    # -------------------------------------------------------------------------
    def predict(self, force_cleanup: bool = False) -> None:

        logging.info('Starting prediction stage')

        # Load model for inference
        model = load_model(
            model_filename=self.conf.model_filename,
            model_dir=self.model_dir
        )

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

        # gather metadata
        if self.conf.metadata_regex is not None:
            metadata = read_metadata(
                self.conf.metadata_regex,
                self.conf.input_bands,
                self.conf.output_bands
            )

        # Gather filenames to predict
        if len(self.conf.inference_regex_list) > 0:
            data_filenames = self.get_filenames(self.conf.inference_regex_list)
        else:
            data_filenames = self.get_filenames(self.conf.inference_regex)
        logging.info(f'{len(data_filenames)} files to predict')

        # iterate files, create lock file to avoid predicting the same file
        for filename in sorted(data_filenames):

            # start timer
            start_time = time.time()

            # set output directory
            output_directory = os.path.join(
                self.conf.inference_save_dir, Path(filename).stem)
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # delete lock file and overwrite prediction if force_cleanup
            logging.warning(
                'You have selected to force cleanup files. ' +
                'This option disables lock file tracking, which' +
                'Could lead to processing the same file multiple times.'
            )
            if force_cleanup and os.path.isfile(lock_filename):
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

                try:

                    logging.info(f'Starting to predict {filename}')

                    # if metadata is available
                    if self.conf.metadata_regex is not None:

                        # get timestamp from filename
                        year_match = re.search(
                            r'(\d{4})(\d{2})(\d{2})', filename)
                        timestamp = str(int(year_match.group(2)))

                        # get monthly values
                        mean = metadata[timestamp]['median'].to_numpy()
                        std = metadata[timestamp]['std'].to_numpy()
                        self.conf.standardization = 'global'

                    # create lock file
                    open(lock_filename, 'w').close()

                    # open filename
                    image = rxr.open_rasterio(filename)
                    logging.info(f'Prediction shape: {image.shape}')

                except rasterio.errors.RasterioIOError:
                    logging.info(f'Skipped {filename}, probably corrupted.')
                    continue

                # Calculate indices and append to the original raster
                logging.info('Adding indices')
                image = indices.add_indices(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)

                # Modify the bands to match inference details
                logging.info('Modifying bands')
                image = modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                # add DTM
                logging.info('Adding DTM layer')
                if image.shape[0] != len(self.conf.output_bands):
                    image = self.add_dtm(filename, image, self.conf.dtm_path)
                    logging.info(f'Prediction shape after modf: {image.shape}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
                # TODO: consider replacing this with the new parameters
                # for better padding.
                temporary_tif = xr.where(image > -100, image, 2000)

                # Sliding window prediction
                prediction = regression_inference.sliding_window_tiler(
                    xraster=temporary_tif,
                    model=model,
                    n_classes=self.conf.n_classes,
                    overlap=self.conf.inference_overlap,
                    batch_size=self.conf.pred_batch_size,
                    standardization=self.conf.standardization,
                    mean=mean,
                    std=std,
                    normalize=self.conf.normalize,
                    window=self.conf.window_algorithm
                ) * self.conf.normalize_label
                prediction[prediction < 0] = 0

                # Drop image band to allow for a merge of mask
                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                )

                # Get metadata to save raster
                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                # Add metadata to raster attributes
                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction.attrs['model_name'] = (self.conf.model_filename)
                prediction = prediction.transpose("band", "y", "x")

                # Add cloudmask to the prediction
                if self.conf.cloudmask_path is not None:

                    # get the corresponding file that matches the
                    # cloudmask regex
                    cloudmask_filename = self.get_filenames(
                        os.path.join(
                            self.conf.cloudmask_path,
                            f'{Path(filename).stem.split("-")[0]}*.tif'
                        ),
                        allow_empty=True
                    )

                    # if we found cloud mask filename, proceed
                    if len(cloudmask_filename) > 0:
                        prediction = self.add_cloudmask(
                            prediction, cloudmask_filename[0])
                    else:
                        logging.info(
                            'No cloud mask filename found, ' +
                            'skipping cloud mask step.')

                # Set nodata values on mask
                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(
                    self.conf.prediction_nodata, encoded=True, inplace=True)

                # Save output raster file to disk
                prediction.rio.to_raster(
                    output_filename,
                    BIGTIFF="IF_SAFER",
                    compress=self.conf.prediction_compress,
                    driver=self.conf.prediction_driver,
                    dtype=self.conf.prediction_dtype
                )
                del prediction

                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

                logging.info(f'Finished processing {output_filename}')
                logging.info(f"{(time.time() - start_time)/60} min")

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')

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
        test_filenames = []
        for data_filename, label_filename in \
                tqdm(zip(test_data_filenames, test_label_filenames)):

            # read data
            test_data_tile = np.moveaxis(
                np.load(data_filename), 0, -1)

            # continue if no-data present
            if np.isnan(test_data_tile).any():
                continue

            # read label
            test_data_label = np.load(label_filename)

            # continue if no-data present
            if np.isnan(test_data_label).any():
                continue

            test_data.append(test_data_tile)
            test_labels.append(test_data_label)
            test_filenames.append(data_filename)

        # if there are no tiles by the end of the iteration
        assert len(test_data) > 0, 'No tiles no validate. Check for no-data.'

        # create array and reshape it
        test_data = np.array(test_data)
        test_labels = np.moveaxis(np.array(test_labels), 1, -1)
        logging.info(f'Reshaped data: {test_data.shape}, {test_labels.shape}')

        # score of how much data is left after no-data filtering
        tile_percentage = test_data.shape[0] / len(test_data_filenames) * 100
        logging.info(
            f'{tile_percentage}% of tiles used for validation after filtering')

        # prediction loop, do prediction in batches of data
        predictions = []
        for batch_i in self._xbatch(
                test_data, batch_size=self.conf.pred_batch_size):

            # standardize
            batch = batch_i.copy()

            if self.conf.standardization is not None:
                for item in range(batch.shape[0]):
                    batch[item, :, :, :] = standardize_image(
                        batch[item, :, :, :],
                        self.conf.standardization, mean, std)

            # predict
            batch = model.predict(
                batch, batch_size=self.conf.pred_batch_size, verbose=1)
            predictions.append(batch)

        # create prediction array
        predictions = np.concatenate(predictions, axis=0)
        logging.info(f'Finished prediction for {predictions.shape[0]} tiles.')

        # perform prediction
        with tf.device("CPU"):
            dataset = Dataset.from_tensor_slices(
                (test_data, test_labels)).batch(self.conf.pred_batch_size)

        logging.info('Starting model evaluation.')
        results = model.evaluate(dataset)

        logging.info("======= TensorFLow Metrics =======")
        logging.info(f'Test Loss: {results[0]}')
        logging.info(f'Test MSE: {results[1]}')
        logging.info(f'Test RMSE: {results[2]}')
        logging.info(f'Test MAE: {results[3]}')
        logging.info(f'Test R2: {results[4]}')

        # squeeze the last dimension of the predictions
        predictions = np.squeeze(predictions)
        test_labels = np.squeeze(test_labels)
        logging.info(f'Shape of predictions {predictions.shape}')

        # Reshape for sckit-learn metrics
        nsamples, nx, ny = predictions.shape
        predictions = predictions.reshape((nsamples, nx * ny))
        test_labels = test_labels.reshape((nsamples, nx * ny))

        # gather some metrics
        logging.info("======= SckitLearn Metrics =======")
        logging.info(f'R2: {r2_score(predictions, test_labels)}')
        logging.info(f'MAE: {mean_absolute_error(predictions, test_labels)}')
        logging.info(f'MSE: {mean_squared_error(predictions, test_labels)}')
        logging.info(
            f'MAPE: {mean_absolute_percentage_error(predictions, test_labels)}'
        )

        # save output tiles
        predictions = predictions.reshape((nsamples, nx, ny))
        logging.info(f'Reshaped predictions back to {predictions.shape}')

        for index, filename in tqdm(enumerate(test_filenames)):
            output_filename = os.path.join(
                self.conf.inference_save_dir,
                f'{Path(filename).stem}.npy'
            )
            np.save(output_filename, predictions[index, :, :])

        logging.info(f'Saved predictions to {self.conf.inference_save_dir}')

        return
