from functools import reduce
import csv

def combine_inputs(to_order):
    pass

def prepare_dataset_data(csvfile, inputs_placement, labels_placement, file_statement_delimiter, training_size_percentage = 0.8):
    dataset = reduce(lambda acc, case: acc + [case], csv.reader(csvfile, delimiter = file_statement_delimiter), [])
    dataset = [list(map(float, row)) for row in dataset]

    dataset_divisor = round(training_size_percentage * len(dataset))

    trainig_dataset = dataset[:dataset_divisor]
    test_dataset = dataset[dataset_divisor:]

    trainig_dataset_inputs = [[row[i] for i in inputs_placement] for row in trainig_dataset]
    test_dataset_inputs = [[row[i] for i in inputs_placement] for row in test_dataset]

    trainig_dataset_labels = [[row[i] for i in labels_placement] for row in trainig_dataset]
    test_dataset_labels = [[row[i] for i in labels_placement] for row in test_dataset]

    return (trainig_dataset_inputs, test_dataset_inputs, trainig_dataset_labels, test_dataset_labels)