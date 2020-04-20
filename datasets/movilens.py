import os
import random

import numpy as np
import pandas as pd


def load_dataset():
    file = os.path.join(os.path.dirname(__file__), 'data/ml-100k/u.data')
    data = pd.read_csv(file, sep='\t', names=['userId', 'movieId', 'rating', 'date'])

    # print(data.iloc[:, :3])  # все строки :, а среди столбцов первые 3 т.е. :3

    # print(np.array(data.iloc[:, :3]))

    data = np.array(data.iloc[:, :3]).tolist()

    # print(data[:1])  # первая строка
    # print(data[99999:])  # последняя строка

    np.random.seed(1234)

    random.shuffle(data)

    train_data = data[:int(len(data) * 0.8)]  # первые 80 тыс строк
    test_data = data[int(len(data) * 0.8):]  # оставщиеся строки т.е. последние 20 тыс. строк
    print('load data finished')
    print('total data ', len(data))
    return train_data, test_data, data


def describe_dataset():
    train_data, test_data, data = load_dataset()
    df = pd.DataFrame(data)
    print(df.iloc[:, 2].describe())
