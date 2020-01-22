# type: ignore[attr-define]

from flambe.nlp.classification.datasets import SSTDataset, TRECDataset, NewsGroupDataset
from flambe.nlp.classification.model import TextClassifier


__all__ = ['TextClassifier', 'SSTDataset', 'TRECDataset', 'NewsGroupDataset']
