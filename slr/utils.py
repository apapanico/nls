
import sys


class DummyFile(object):
    def write(self, x):
        pass


def nostdout(f):

    def wrapper(*args, **kwargs):
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        out = f(*args, **kwargs)
        sys.stdout = save_stdout
        return out

    return wrapper
