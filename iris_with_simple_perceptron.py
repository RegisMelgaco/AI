import csv
from codes.simple_perceptron import PerceptronNeuron
from functools import reduce
from random import shuffle


with open("datasets/iris.csv", newline="\n") as csvfile:
    irirs_dataset = reduce(lambda acc, case: acc + [case], csv.reader(csvfile, delimiter = ","), [])
    shuffle(irirs_dataset)
    dataset_input, dataset_labels = [], []

    for [sepal_length, sepal_width, petal_length, petal_width, spicie] in irirs_dataset:
        dataset_input.append(list(map(float, [sepal_length, sepal_width, petal_length, petal_width])))
        dataset_labels.append(1 if spicie == "Iris-setosa" else 0)
    
    dataset_input_size = len(dataset_input[0])
    dataset_size = len(dataset_input)

    trainig_dataset_input, trainig_dataset_labels = dataset_input[:int(dataset_size*0.8)], dataset_labels[:int(dataset_size*0.8)]
    test_dataset_input, test_dataset_labels = dataset_input[int(dataset_size*0.8):], dataset_labels[int(dataset_size*0.8):]

    sp = PerceptronNeuron(dataset_input_size, 0.1, 10, -10)
    sp.fit(trainig_dataset_input, trainig_dataset_labels)

    correct_cases = 0
    for (input, label) in zip(test_dataset_input, test_dataset_labels):
        correct_cases += 1 if label == sp.predict(input) else 0

    print("taxa de acerto:", correct_cases/len(test_dataset_input))
    print("contagem de eras:", sp.age)