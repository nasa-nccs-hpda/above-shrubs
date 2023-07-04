from typing import Optional
from dataclasses import dataclass
from tensorflow_caney.model.config.cnn_config import Config


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
