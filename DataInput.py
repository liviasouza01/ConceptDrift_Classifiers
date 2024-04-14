from Datas import Datasets
from skmultiflow.data import DataStream
from sklearn.model_selection import train_test_split


Data = Datasets()
X, y = Data.sea() #example

'''
sine()
stagger()
'''

stream = DataStream(X, y)
print(stream.next_sample(100))



