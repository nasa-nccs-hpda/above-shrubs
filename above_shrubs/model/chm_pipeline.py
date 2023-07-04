import os
import re
import time
import logging
import rasterio
import numpy as np
import xarray as xr
from tqdm import tqdm
import rioxarray as rxr
from pathlib import Path
from itertools import repeat
from pygeotools.lib import iolib, warplib
from multiprocessing import Pool, cpu_count

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
        self.logger.info(f'Model dir: {self.model_dir}')

        # Create output directories
        for out_dir in [
                self.images_dir, self.labels_dir,
                self.metadata_dir, self.model_dir]:
            os.makedirs(out_dir, exist_ok=True)

        # Seed everything
        seed_everything(self.conf.seed)

    # -------------------------------------------------------------------------
    # add_dtm
    # -------------------------------------------------------------------------
    def add_dtm(self, raster_filename, raster, dtm_filename):

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
            dtm.any(dim = 'band') != True, int(raster.rio.nodata))
        dtm = dtm.where(dtm > 0, int(raster.rio.nodata))
        dtm.attrs['long_name'] = dtm.attrs['long_name'] + ("DTM",)
        return dtm

    def add_cloudmask(self, raster, cloudmask_filename):
        cloud_raster = rxr.open_rasterio(cloudmask_filename)
        raster = raster.where(cloud_raster[0, :, :] == 0, -9999)
        return raster

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
    def predict(self) -> None:

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
            basename = os.path.basename(os.path.dirname(filename))
            if basename == 'M1BS' or basename == 'P1BS':
                basename = os.path.basename(
                    os.path.dirname(os.path.dirname(filename)))

            output_directory = os.path.join(
                self.conf.inference_save_dir, basename)
            os.makedirs(output_directory, exist_ok=True)

            # set prediction output filename
            output_filename = os.path.join(
                output_directory,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif')

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

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
                image = self.add_dtm(filename, image, self.conf.dtm_path)
                logging.info(f'Prediction shape after modf: {image.shape}')

                # Transpose the image for channel last format
                image = image.transpose("y", "x", "band")

                # Remove no-data values to account for edge effects
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
                    cloudmask_filename = self.get_filenames(
                        os.path.join(
                            self.conf.cloudmask_path,
                            f'{Path(filename).stem.split("-")[0]}*.tif'
                        )
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

        # gather some metrics
        # RMSE
        # R2
        # MAE

        # save output tiles
        for index, filename in tqdm(enumerate(test_data_filenames)):

            output_filename = os.path.join(
                self.conf.inference_save_dir,
                f'{Path(filename).stem}.npy'
            )
            np.save(output_filename, predictions[index, :, :])

        return
