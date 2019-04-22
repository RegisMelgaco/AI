import csv
from functools import reduce
from random import shuffle

from codes.adaline import fit, predict


def prepare_dataset_data(dataset, training_size_percentage = 0.8):
    dataset_divisor = round(training_size_percentage * len(dataset))

    trainig_dataset = dataset[:dataset_divisor]
    test_dataset = dataset[dataset_divisor:]
    
    trainig_dataset_inputs = reduce(lambda acc, case: acc + [case[:-1]], trainig_dataset, [])
    test_dataset_inputs = reduce(lambda acc, case: acc + [case[:-1]], test_dataset, [])
    trainig_dataset_labels = reduce(lambda acc, case: acc + [case[-1]], trainig_dataset, [])
    test_dataset_labels = reduce(lambda acc, case: acc + [case[-1]], test_dataset, [])

    return (trainig_dataset_inputs, test_dataset_inputs, trainig_dataset_labels, test_dataset_labels)

print("=== red wine ===")
with open("datasets/winequality-red.csv", newline="\n") as csvfile:
    untreated_dataset = reduce(lambda acc, case: acc + [case], csv.reader(csvfile, delimiter = ";"), [])
    dataset = [list(map(float, row)) for row in untreated_dataset]
    num_of_tests = 30
    error_history = []

    for i in range(num_of_tests):
        print("test number:", i+1)
        shuffle(dataset)
        (trainig_dataset_inputs,test_dataset_inputs, trainig_dataset_labels, test_dataset_labels) = prepare_dataset_data(dataset)

        test_dataset_labels_len = len(test_dataset_labels)

        weights = fit(test_dataset_inputs, test_dataset_labels, 0.00005, 5000)
        # print(weights)

        predictions = [predict(input, weights) for input in test_dataset_inputs]
        errors = [abs(l - p) for (p, l) in zip(predictions, test_dataset_labels)]
        average_error = sum(errors)/len(errors)
        error_history.append(average_error)

    print("==================================")

    average_accurace = sum(error_history) / len(error_history)
    standard_deviation = reduce(lambda acc, accuracy: acc + abs(average_accurace-accuracy), error_history, 0)/len(error_history)

    print("number of tests:", num_of_tests)
    print("average accurace:", average_accurace)
    print("standard deviation:", standard_deviation)


print("=== white wine ===")
with open("datasets/winequality-white.csv", newline="\n") as csvfile:
    untreated_dataset = reduce(lambda acc, case: acc + [case], csv.reader(csvfile, delimiter = ";"), [])
    dataset = [list(map(float, row)) for row in untreated_dataset]
    num_of_tests = 30
    error_history = []

    for i in range(num_of_tests):
        print("test number:", i+1)
        shuffle(dataset)
        (trainig_dataset_inputs,test_dataset_inputs, trainig_dataset_labels, test_dataset_labels) = prepare_dataset_data(dataset)

        test_dataset_labels_len = len(test_dataset_labels)

        weights = fit(test_dataset_inputs, test_dataset_labels, 0.00005, 50000)
        # print(weights)

        predictions = [predict(input, weights) for input in test_dataset_inputs]
        errors = [abs(l - p) for (p, l) in zip(predictions, test_dataset_labels)]
        average_error = sum(errors)/len(errors)
        error_history.append(average_error)

    print("==================================")

    average_accurace = sum(error_history) / len(error_history)
    standard_deviation = reduce(lambda acc, accuracy: acc + abs(average_accurace-accuracy), error_history, 0)/len(error_history)

    print("number of tests:", num_of_tests)
    print("average accurace:", average_accurace)
    print("standard deviation:", standard_deviation)