from data_utils import DataSet
from StringIO import StringIO


class SpamAssasin(DataSet):

    main_site = 'https://spamassassin.apache.org/publiccorpus/'

    resources = {
        'easy_ham': '20021010_easy_ham.tar.bz2',
        'hard_ham': '20021010_hard_ham.tar.bz2',
        'spam': '20021010_spam.tar.bz2',
        'easy_ham_2': '20030228_easy_ham.tar.bz2',
        'easy_ham_3': '20030228_easy_ham_2.tar.bz2',
        'hard_ham_2': '20030228_hard_ham.tar.bz2',
        'spam_2': '20030228_spam.tar.bz2',
        'spam_3': '20030228_spam_2.tar.bz2',
        'spam_4': '20050311_spam_2.tar.bz2',

    }


    def __init__(self):
        super(DataSet, self).__init__()





