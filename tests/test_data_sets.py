import unittest

from bayes.datasets import get_tarfile


class TestDataSets(unittest.TestCase):

    def test_download(self):
        raw_data = get_tarfile('https://spamassassin.apache.org/publiccorpus/20050311_spam_2.tar.bz2')
        #print raw_data.readlines()




if __name__ == '__main__':
    unittest.main()

