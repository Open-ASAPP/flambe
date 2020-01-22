import os
import logging


class ContextualFileHandler(logging.FileHandler):
    """Uses the record `current_log_dir` value to customize file path

    Uses the LogRecord object's current_log_dir value to dynamically
    determine a path for the output file name. Functions
    the same as parent `logging.FileHandler` but always writes to
    the file given by `current_log_dir + canonical_name`.

    Parameters
    ----------
    canonical_name : str
        Common name for each file
    mode : str
        See built-in `open` description of `mode`
    encoding : type
        See built-in `open` description of `encoding`

    Attributes
    ----------
    current_log_dir : str
        Most recently used prefix in an incoming `LogRecord`
    canonical_name : str
        Common name for each file
    mode : str
        See built-in `open` description of `mode`
    delay : type
        If true will delay opening of file until first use
    stream : type
        Currently open file stream for writing logs - should match
        the file indicated by `base_path` + `current_prefix` +
        `canonical_name`
    encoding
        See built-in `open` description of `encoding`

    """

    def __init__(self,  # type: ignore
                 canonical_name: str,
                 mode: str = 'a',
                 encoding=None) -> None:
        self.current_log_dir = ""
        self.canonical_name = canonical_name
        self.mode = mode
        self.encoding = encoding
        self.delay = True
        logging.Handler.__init__(self)
        self.stream = None  # type: ignore

    @property
    def baseFilename(self) -> str:  # type: ignore
        """Output filename; Override parent property to use prefix"""
        return os.path.join(self.current_log_dir, self.canonical_name)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record

        If the stream is invalidated by a new record `prefix` value
        it will be closed and set to `None` before calling the super
        `emit` which will handle opening a new stream to `baseFilename`

        Parameters
        ----------
        record : logging.LogRecord
            Record to be saved at `_console_log_dir`

        Returns
        -------
        None

        """
        if self.current_log_dir != record._console_log_dir:  # type: ignore
            self.current_log_dir = record._console_log_dir  # type: ignore
            # Close current stream if it exists
            self.close()
            # Ensure stream is `None` in case close failed
            self.stream = None  # type: ignore
            new_path = self.current_log_dir
            if not os.path.isdir(new_path):
                os.makedirs(new_path)
        super().emit(record)
