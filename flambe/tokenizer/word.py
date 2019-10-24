
from typing import Union, List, Optional

import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flambe.tokenizer import Tokenizer


class WordTokenizer(Tokenizer):
    """Implement a word level tokenizer using
       nltk.tokenize.word_tokenize """

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output word tokens, as a list of strings

        """
        return example.split()


class NLTKWordTokenizer(Tokenizer):
    """Implement a word level tokenizer using
       nltk.tokenize.word_tokenize """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nltk.download('punkt', quiet=True)

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output word tokens, as a list of strings

        """
        return word_tokenize(example)


class NGramsTokenizer(Tokenizer):
    """Implement a n-gram tokenizer

    Examples
    --------

    >>> t = NGramsTokenizer(ngrams=2).tokenize("hi how are you?")
    ['hi, how', 'how are', 'are you?']

    >>> t = NGramsTokenizer(ngrams=[1,2]).tokenize("hi how are you?")
    ['hi,', 'how', 'are', 'you?', 'hi, how', 'how are', 'are you?']

    Parameters
    ----------
    ngrams: Union[int, List[int]]
        An int or a list of ints. If it's a list of ints, all n-grams
        (for each int) will be considered in the tokenizer.
    exclude_stopwords: bool
        Whether to exlude stopword or not.
        See the related param stop_words
    stop_words: Optional[List]
        List of stop words to exclude when exclude_stopwords is True.
        If None set to nltk.corpus.stopwords.

    """
    def __init__(self, ngrams: Union[int, List[int]] = 1,
                 exclude_stopwords: bool = False,
                 stop_words: Optional[List] = None) -> None:
        """ Initialize the NGramsTokenizer

        Parameters
        ----------
        ngrams : Union[int, List[int]], optional
            [description], by default 1
        exclude_stopwords: bool
            [description], by default False
        stop_words: Optional[List]
            [description], by default None

        """
        self.ngrams = ngrams
        self.exclude_stopwords = exclude_stopwords

        if self.exclude_stopwords:
            self.stop_words = stop_words
            if self.stop_words is None:
                nltk.download('stopwords', quiet=True)
                self.stop_words = stopwords.words('english')

        nltk.download('punkt', quiet=True)

    @staticmethod
    def _tokenize(example: str, n: int) -> List[str]:
        """Tokenize an input example using ngrams.

        """
        return list(" ".join(x) if len(x) > 1 else x[0] for x in ngrams(word_tokenize(example), n))

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string.

        Returns
        -------
        List[str]
            The output word tokens, as a list of strings

        """
        if self.exclude_stopwords and self.stop_words:
            example = ' '.join([word for word in word_tokenize(example)
                                if word not in self.stop_words])

        if isinstance(self.ngrams, List):
            ret: List[str] = []
            for i in self.ngrams:
                ret.extend(self._tokenize(example, i))
            return ret
        else:
            return NGramsTokenizer._tokenize(example, self.ngrams)
