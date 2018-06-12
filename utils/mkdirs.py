import errno
import os

def mkdirs(path):
    """
    Recursively create directories with the given path.
    """
    try:
        os.makedirs(path)
    except OSError as ex:
        if not (ex.errno == errno.EEXIST and os.path.isdir(path)):
            raise