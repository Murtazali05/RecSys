import csv
import os
import urllib
import zipfile
from os.path import dirname
from os.path import join

import numpy as np

URL = "http://www.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
ARCHIVE_NAME = "BX-CSV-Dump.zip"


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def download_book_crossings(target_dir):
    """ Download the book-crossing data and unzip it """
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        opener = urllib.urlopen(URL)
        open(archive_path, 'wb').write(opener.read())

    source_zip = zipfile.ZipFile(archive_path, 'r')
    archives = []
    for name in source_zip.namelist():
        if name.find('.csv') != -1:
            source_zip.extract(name, target_dir)
            archives.append(name)
    source_zip.close()
    os.remove(archive_path)

    return archives


def load_bookcrossings(data_home=None, download_if_missing=True, implicit=False):
    if data_home:
        if not os.path.exists(data_home):
            os.makedirs(data_home)
    else:
        data_home = join(dirname(__file__), 'data/BX-CSV-Dump')

    try:
        if not os.path.exists(os.path.join(data_home, 'BX-Book-Ratings.csv')) \
                and not open(os.path.join(data_home, 'BX-Books.csv')):
            raise IOError
    except Exception as e:
        print(80 * '_')
        print('Loading files failed')
        print(80 * '_')
        print(e)

        if download_if_missing:
            print('downloading the dataset...')
            try:
                download_book_crossings(data_home)
            except:
                raise Exception('FAIL: Problems during the download.')
            print('dataset downloaded.')
        else:
            raise IOError('Book-Crossing dataset not found')

    ratings_m = csv.reader(open(os.path.join(data_home,
                                             'BX-Book-Ratings.csv')), delimiter=';')
    ratings_m.next()
    data_books = {}
    if implicit:
        for user_id, item_id, rating in ratings_m:
            if rating == "0":
                data_books.setdefault(user_id, {})
                data_books[user_id][item_id] = True
    else:
        for user_id, item_id, rating in ratings_m:
            rating = int(rating)
            if rating != "0":
                data_books.setdefault(user_id, {})
                data_books[user_id][item_id] = int(rating)

    # Read the titles
    data_titles = np.loadtxt(os.path.join(data_home, 'BX-Books.csv'),
                             delimiter=';', usecols=(0, 1), dtype=str)

    data_t = []
    for item_id, label in data_titles:
        data_t.append((item_id, label))
    data_titles = dict(data_t)

    fdescr = open(dirname(__file__) + '/descr/book-crossing.rst')

    return Bunch(data=data_books, item_ids=data_titles,
                 user_ids=None, DESCR=fdescr.read())
