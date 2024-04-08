from DataSGeneration import DataStreamGenerator
from skmultiflow.data import DataStream

C = DataStreamGenerator(class_count=2, attribute_count=2, sample_count=100000, noise=True, redunce_variable=True)
X, Y = C.Linear_Conditions(plot=True, save=True)
stream = DataStream(X, Y)
print(stream.next_sample(100))