import csv
from functools import reduce
from random import shuffle

from sklearn.metrics import confusion_matrix

from codes.uni_layer_perceptron import fit, predict


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
    classes = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    iris_dataset = list(map(lambda row: row[:-1]+[classes[row[-1]]], untreated_irirs_dataset))
    iris_dataset = [list(map(float, row)) for row in iris_dataset]
    confusion_matrix_history = []
    test_dataset_labels_len = 0
    num_of_tests = 30

    for i in range(num_of_tests):
        print("treinamento num:", i)
        shuffle(iris_dataset)
        (trainig_dataset_inputs,test_dataset_inputs, trainig_dataset_labels, test_dataset_labels) = prepare_dataset_data(iris_dataset)

        test_dataset_labels_len = len(test_dataset_labels)

        weights_list = fit(test_dataset_inputs, test_dataset_labels, 3, 0.005, 10000)

        predictions = [predict(input, weights_list) for input in test_dataset_inputs]
        cf = confusion_matrix(test_dataset_labels, predictions)
        confusion_matrix_history.append(cf)

    matrix_diagonal = lambda matrix: [matrix[i][i] for i in range(len(matrix))]
    calc_accuracy = lambda confusion_matrix, cases_count: sum(matrix_diagonal(confusion_matrix))/cases_count

    accuracy_history = [calc_accuracy(cm, test_dataset_labels_len) for cm in confusion_matrix_history]
    average_accurace = sum(accuracy_history) / len(accuracy_history)
    standard_deviation = reduce(lambda acc, accuracy: acc + abs(average_accurace-accuracy), accuracy_history, 0)/len(accuracy_history)

    print("number of confusion matrixes:", len(confusion_matrix_history))
    print("average accurace:", average_accurace)
    print("standard deviation:", standard_deviation)
