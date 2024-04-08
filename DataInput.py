from Datas import Datasets
from skmultiflow.data import DataStream
from sklearn.model_selection import train_test_split


Data = Datasets()
X, y = Data.sea() #example

'''
sine()
stagger()
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

stream = DataStream(X_train, y_train)
print(stream.next_sample(100))



