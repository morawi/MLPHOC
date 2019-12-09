import logging
import sys
import contextlib

def setup_logging(log_path=None, log_level=logging.INFO, write_stdout=True):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(name)s:%(lineno)d %(levelname)s :: %(message)s')

    # Create console handler to write to stdout
    if write_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_path is not None:
        # Create file handler, attach formatter and add it to the logger
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


@contextlib.contextmanager
def log_context(log_path, log_level=logging.INFO, write_stdout=True):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(name)s:%(lineno)d %(levelname)s :: %(message)s')

    # Create console handler to write to stdout
    if write_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        stream_handler = None

    if log_path is not None:
        # Create file handler, attach formatter and add it to the logger
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = None

    yield

    # Clean up root logger
    if stream_handler is not None:
        logger.removeHandler(stream_handler)
    if file_handler is not None:
        logger.removeHandler(file_handler)