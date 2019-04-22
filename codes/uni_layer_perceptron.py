# simple perceptron prediction of multiple classes

from random import uniform, shuffle
from functools import reduce


def activation (inputs, weights):
    inputs_with_bias = inputs + [1]
    acc = 0
    for (input, weight) in zip(inputs_with_bias, weights):
        acc += input*weight
    return acc

def predict (inputs, weights_list):
    activations = [activation(inputs, w) for w in weights_list]
    return activations.index(sorted(activations)[-1])

def fit (train_payload, labels, classes_count, learning_rate, max_eras = 10000):
    num_inputs = len(train_payload[0])
    weights_list = [gen_random_weights(num_inputs+1) for i in range(classes_count)]

    era = 0
    got_all = False
    while era < max_eras and not got_all:
        got_all = True
        
        for (inputs, label) in zip(train_payload, labels):
            errors = calc_errors(inputs, weights_list, label, classes_count)

            if (errors.count(0) != len(errors)):
                got_all = False
                # print("a:", weights_list)
                weights_list = [update_weights(weights, inputs, error, learning_rate) for (weights, error) in zip(weights_list, errors)]
                # print("d:", weights_list)
        era += 1

    return weights_list

def gen_random_weights(num_weights):
    return [uniform(-1.0, 1.0) for i in range(num_weights)]

def calc_errors(inputs, weights_list, label, classes_count):
    activations = [activation(inputs, weights) for weights in weights_list]
    predicts = [0 if act < 0 else 1 for act in activations]
    labels = [1 if label == i else 0 for i in range(classes_count)]
    return [l - p for (l, p) in zip(labels, predicts)]

def update_weights(weights, inputs, error, learning_rate):
    return [weight + error * learning_rate * input for (weight, input) in zip(weights, inputs+[1])]
