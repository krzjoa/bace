import wget
from StringIO import StringIO
import sys


class DataSet(object):
    ''''''

    # def __init__(self, data, targets):
    #     self.data = data
    #     self.targets = targets

    # def __len__(self):
    #     return len(self.data)

def download(url):
    sys.stdout = sys.__stdout__
    raw_data = StringIO()
    raw_data.write(wget.download(url))
    return raw_data

def get_tarfile(url):
    from tarfile import TarFile
    raw_data = download(url)
    return TarFile(raw_data)