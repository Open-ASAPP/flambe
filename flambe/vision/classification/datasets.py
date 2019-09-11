import gzip
import torch
import requests
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split

import numpy as np

from flambe.dataset import Dataset
from flambe.compile import registrable_factory


class MNISTDataset(Dataset):
    """The official MNIST dataset."""

    data_type = {
        0x08: np.uint8,
        0x09: np.int8,
        0x0b: np.dtype('>i2'),
        0x0c: np.dtype('>i4'),
        0x0d: np.dtype('>f4'),
        0x0e: np.dtype('>f8'),
    }

    URL = "http://yann.lecun.com/exdb/mnist/"

    def __init__(self,
                 train_images: np.ndarray = None,
                 train_labels: np.ndarray = None,
                 test_images: np.ndarray = None,
                 test_labels: np.ndarray = None,
                 val_ratio: Optional[float] = 0.2,
                 seed: Optional[int] = None) -> None:
        """Initialize the MNISTDataset.

        Parameters
        ----------
        train_images: np.ndarray
            parsed train images as a numpy array
        train_labels: np.ndarray
            parsed train labels as a numpy array
        test_images: np.ndarray
            parsed test images as a numpy array
        test_labels: np.ndarray
            parsed test labels as a numpy array
        val_ratio: Optional[float]
            validation set ratio. Default 0.2
        seed: Optional[int]
            random seed for the validation set split
        """
        if train_images is None:
            train_images = self._parse_downloaded_idx(self.URL + "train-images-idx3-ubyte.gz")
        if train_labels is None:
            train_labels = self._parse_downloaded_idx(self.URL + "train-labels-idx1-ubyte.gz")
        if test_images is None:
            test_images = self._parse_downloaded_idx(self.URL + "t10k-images-idx3-ubyte.gz")
        if test_labels is None:
            test_labels = self._parse_downloaded_idx(self.URL + "t10k-labels-idx1-ubyte.gz")

        self.train_images, self.val_images, self.train_labels, self.val_labels = \
            train_test_split(train_images, train_labels, test_size=val_ratio, random_state=seed)
        self.test_images = test_images
        self.test_labels = test_labels

        self._train = get_dataset(self.train_images, self.train_labels)
        self._val = get_dataset(self.val_images, self.val_labels)
        self._test = get_dataset(self.test_images, self.test_labels)

    @registrable_factory
    @classmethod
    def from_path(cls,
                  train_images_path: str,
                  train_labels_path: str,
                  test_images_path: str,
                  test_labels_path: str,
                  val_ratio: Optional[float] = 0.2,
                  seed: Optional[int] = None) -> 'MNISTDataset':
        """Initialize the MNISTDataset from local files.

        Parameters
        ----------
        train_images_path: str
            path to the train images file in the idx format
        train_labels_path: str
            path to the train labels file in the idx format
        test_images_path: str
            path to the test images file in the idx format
        test_labels_path: str
            path to the test labels file in the idx format
        val_ratio: Optional[float]
            validation set ratio. Default 0.2
        seed: Optional[int]
            random seed for the validation set split
        """
        return cls(
            cls._parse_local_gzipped_idx(train_images_path),
            cls._parse_local_gzipped_idx(train_labels_path),
            cls._parse_local_gzipped_idx(test_images_path),
            cls._parse_local_gzipped_idx(test_labels_path),
            val_ratio=val_ratio,
            seed=seed,
        )

    @property
    def train(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the training data"""
        return self._train

    @property
    def val(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the validation data"""
        return self._val

    @property
    def test(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the test data"""
        return self._test

    @classmethod
    def _parse_local_gzipped_idx(cls, path: str) -> np.ndarray:
        """Parse a local gzipped idx file"""
        with gzip.open(path) as f:
            return cls._parse_idx(f.read())

    @classmethod
    def _parse_downloaded_idx(cls, url: str) -> np.ndarray:
        """Parse a downloaded idx file"""
        r = requests.get(url)
        return cls._parse_idx(gzip.decompress(r.content))

    @classmethod
    def _parse_idx(cls, data: bytes) -> np.ndarray:
        """Parse an idx filie"""
        # parse the magic number that contains the dimension
        # and the data type
        magic = int.from_bytes(data[0:4], 'big')
        dim = magic % 256
        data_type = magic // 256

        shape = [int.from_bytes(data[4 * (i + 1): 4 * (i + 2)], 'big') for i in range(dim)]
        return np.frombuffer(
            data,
            dtype=cls.data_type[data_type],
            offset=4 * (dim + 1)
        ).reshape(shape)


def get_dataset(images: np.ndarray, labels: np.ndarray) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(
        torch.from_numpy(image).float().unsqueeze(0),
        torch.tensor(label).long()
    ) for image, label in zip(images, labels)]
