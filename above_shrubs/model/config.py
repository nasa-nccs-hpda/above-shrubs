from typing import Optional
from dataclasses import dataclass
from tensorflow_caney.model.config.cnn_config import Config as TFC_Config


@dataclass
class Config(TFC_Config):

    # make data_dir optional for this iteration
    data_dir: Optional[str] = None

    # model directory to store model at
    model_dir: Optional[str] = None

    # project name specified for the configuration
    project_name: Optional[str] = 'above_shrubs'

    # model name to use
    model_name: Optional[str] = 'custom_unet'

    # version of the model to use
    version: Optional[str] = '1.1.1'

    # main directory to use for top level directory
    main_dir: Optional[str] = 'output'

    # directory where the original training and testing tiles are at
    train_tiles_dir: Optional[str] = None
    test_tiles_dir: Optional[str] = None


@dataclass
class CHMConfig(Config):

    # directory that store train data
    train_data_dir: Optional[str] = None
    train_label_dir: Optional[str] = None

    # directory that stores test data
    test_dir: Optional[str] = None
    test_data_dir: Optional[str] = None
    test_label_dir: Optional[str] = None

    # filenames storing DTM and DSM
    dtm_path: Optional[str] = None
    dsm_path: Optional[str] = None

    # filenames storing cloud mask
    cloudmask_path: Optional[str] = None


@dataclass
class LandCoverConfig(Config):

    # directory that store train data
    train_data_dir: Optional[str] = None
    train_label_dir: Optional[str] = None

    # directory that stores test data
    test_dir: Optional[str] = None
    test_data_dir: Optional[str] = None
    test_label_dir: Optional[str] = None

    # filenames storing DTM and DSM
    dtm_path: Optional[str] = None
    dsm_path: Optional[str] = None

    # filenames storing cloud mask
    cloudmask_path: Optional[str] = None
    cloudmask_value: Optional[int] = None

    # regex to find CHMs (cannot use path because
    # of inference directory structure)
    chm_regex: Optional[str] = None

    # set probability to False by default
    probability_map: Optional[bool] = False
