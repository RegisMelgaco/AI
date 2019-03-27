import csv
from functools import reduce
from random import shuffle

from sklearn.metrics import confusion_matrix

from codes.simple_perceptron import PerceptronNeuron


# last col must be the label
def prepare_dataset_data(dataset, training_size_percentage = 0.8):
    dataset_divisor = round(training_size_percentage * len(dataset))
    trainig_dataset = dataset[:dataset_divisor]
    test_dataset = dataset[dataset_divisor:]
    
    trainig_dataset_inputs = reduce(lambda acc, case: acc + [case[:-1]], trainig_dataset, [])
    test_dataset_inputs = reduce(lambda acc, case: acc + [case[:-1]], test_dataset, [])
    trainig_dataset_labels = reduce(lambda acc, case: acc + [case[-1]], trainig_dataset, [])
    test_dataset_labels = reduce(lambda acc, case: acc + [case[-1]], test_dataset, [])

    return (trainig_dataset_inputs, test_dataset_inputs, trainig_dataset_labels, test_dataset_labels)

with open("datasets/iris.csv", newline="\n") as csvfile:
    untreated_irirs_dataset = reduce(lambda acc, case: acc + [case], csv.reader(csvfile, delimiter = ","), [])
    iris_dataset = list(map(lambda row: row[:-1]+[1] if row[-1] == "Iris-setosa" else row[:-1]+[0], untreated_irirs_dataset))
    iris_dataset = [list(map(float, row)) for row in iris_dataset]
    confusion_matrix_history = []
    test_dataset_labels_len = 0
    num_of_tests = 1000

    for i in range(num_of_tests):
        shuffle(iris_dataset)
        (trainig_dataset_inputs,test_dataset_inputs, trainig_dataset_labels, test_dataset_labels) = prepare_dataset_data(iris_dataset)

        test_dataset_labels_len = len(test_dataset_labels)

        neuron = PerceptronNeuron(len(trainig_dataset_inputs[0]), 0.0001, 10, -10, 100000)
        neuron.fit(trainig_dataset_inputs, trainig_dataset_labels)
        predictions = [neuron.predict(input) for input in test_dataset_inputs]
        confusion_matrix_history.append(confusion_matrix(test_dataset_labels, predictions))

    matrix_diagonal = lambda matrix: [matrix[i][i] for i in range(len(matrix))]
    calc_accuracy = lambda confusion_matrix, cases_count: sum(matrix_diagonal(confusion_matrix))/cases_count

    accuracy_history = [calc_accuracy(cm, test_dataset_labels_len) for cm in confusion_matrix_history]
    average_accurace = sum(accuracy_history) / len(accuracy_history)
    standard_deviation = reduce(lambda acc, accuracy: acc + abs(average_accurace-accuracy), accuracy_history, 0)/len(accuracy_history)

    print("number of confusion matrixes:", len(confusion_matrix_history))
    print("average accurace:", average_accurace)
    print("standard deviation:", standard_deviation)
