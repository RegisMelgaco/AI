import csv
from codes.simple_perceptron import PerceptronNeuron
from functools import reduce
from random import shuffle


with open("datasets/iris.csv", newline="\n") as csvfile:
    irirs_dataset = reduce(lambda acc, case: acc + [case], csv.reader(csvfile, delimiter = ","), [])
    shuffle(irirs_dataset)
    dataset_input = []
    dataset_labels = {"setosa": [], "versicolor": [], "virginica": []}

    for [sepal_length, sepal_width, petal_length, petal_width, spicie] in irirs_dataset:
        dataset_input.append(list(map(float, [sepal_length, sepal_width, petal_length, petal_width])))

        dataset_labels["setosa"].append(1 if spicie == "Iris-setosa" else 0)
        dataset_labels["versicolor"].append(1 if spicie == "Iris-versicolor" else 0)
        dataset_labels["virginica"].append(1 if spicie == "Iris-virginica" else 0)

    dataset_input_size = len(dataset_input[0])
    dataset_size = len(dataset_input)

    test_dataset_input = dataset_input[int(dataset_size*0.2):]
    test_dataset_labels = {"setosa": dataset_labels["setosa"][int(dataset_size*0.2):],
                           "versicolor": dataset_labels["versicolor"][int(dataset_size*0.2):],
                           "virginica": dataset_labels["virginica"][int(dataset_size*0.2):]}

    trainig_dataset_input = dataset_input[:int(dataset_size*0.8)]
    trainig_dataset_labels = {"setosa": dataset_labels["setosa"][:int(dataset_size*0.8)],
                              "versicolor": dataset_labels["versicolor"][:int(dataset_size*0.8)],
                              "virginica": dataset_labels["virginica"][:int(dataset_size*0.8)]}

    neurons = {"setosa": PerceptronNeuron(dataset_input_size, 0.001, 10, -10, 1000000),
               "versicolor": PerceptronNeuron(dataset_input_size, 0.001, 10, -10, 1000000),
               "virginica": PerceptronNeuron(dataset_input_size, 0.001, 10, -10, 1000000)}

    neurons["setosa"].fit(trainig_dataset_input, trainig_dataset_labels["setosa"])
    neurons["versicolor"].fit(trainig_dataset_input, trainig_dataset_labels["versicolor"])
    neurons["virginica"].fit(trainig_dataset_input, trainig_dataset_labels["virginica"])


    correct_cases = 0
    for (input, label) in zip(test_dataset_input, test_dataset_labels["setosa"]):
        correct_cases += 1 if label == neurons["setosa"].predict(input) else 0

    print("taxa de acerto:", correct_cases/len(test_dataset_input))
    print("contagem de eras:", neurons["setosa"].age)

    correct_cases = 0
    for (input, label) in zip(test_dataset_input, test_dataset_labels["versicolor"]):
        correct_cases += 1 if label == neurons["versicolor"].predict(input) else 0

    print("taxa de acerto:", correct_cases/len(test_dataset_input))
    print("contagem de eras:", neurons["versicolor"].age)

    correct_cases = 0
    for (input, label) in zip(test_dataset_input, test_dataset_labels["virginica"]):
        correct_cases += 1 if label == neurons["virginica"].predict(input) else 0

    print("taxa de acerto:", correct_cases/len(test_dataset_input))
    print("contagem de eras:", neurons["virginica"].age)
