import numpy as np
from tensorflow_caney.model.dataloaders.regression import RegressionDataLoader
from tensorflow_caney.utils.data import get_mean_std_metadata, \
    standardize_image, normalize_image, normalize_meanstd


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
