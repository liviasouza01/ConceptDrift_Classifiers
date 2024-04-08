import pandas as pd

class Datasets(object):
    def __init__(self):
        return

    def Read_Csv(self, url):
        data = pd.read_csv(url)
        data = data.values
        l, w = data.shape
        X = data[:, 0 : w - 1]
        y = data[:, w - 1]
        return X, y

    def sea(self):
        return Datasets.Read_Csv(self,
                                 '.Datasets/Synthetic/Harvard Datasets/sea_0123_abrupto_noise_0.2.csv')

    def sine(self):
        return Datasets.Read_Csv(self,
                                 '.Datasets/Synthetic/Harvard Datasets/sine_0123_abrupto.csv')

    def stagger(self):
        return Datasets.Read_Csv(self,
                                 '.Datasets/Synthetic/Harvard Datasets/stagger_0120_abrupto.csv')
