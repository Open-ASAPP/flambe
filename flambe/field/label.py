from typing import Optional
import torch

from flambe.field.text import TextField
from flambe.tokenizer import LabelTokenizer


class LabelField(TextField):
    """Featurizes input labels.

    The class also handles multilabel inputs and one hot encoding.

    """
    def __init__(self,
                 one_hot: bool = False,
                 multilabel_sep: Optional[str] = None) -> None:
        """Initializes the LabelFetaurizer.

        Parameters
        ----------
        one_hot : bool, optional
            Set for one-hot encoded outputs, defaults to False
        multilabel_sep : str, optional
            If given, splits the input label into multiple labels using
            the given separator, defaults to None.

        """
        self.one_hot = one_hot
        self.multilabel_sep = multilabel_sep

        tokenizer = LabelTokenizer(self.multilabel_sep)
        super().__init__(tokenizer=tokenizer,
                         unk_token=None,
                         pad_token=None)

    def process(self, example):
        """Featurize a single example.

        Parameters
        ----------
        example: str
            The input label

        Returns
        -------
        torch.Tensor
            A list of integer tokens

        """
        # First process normally
        n = super().process(example)

        if self.one_hot:
            n = [int(i in n) for i in range(len(self.vocab))]
            n = torch.tensor(n).long()  # Back to Tensor

        return n
