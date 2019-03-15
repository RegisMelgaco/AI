from random import randint, shuffle
from functools import reduce
import csv
import os



class PerceptronNeuron:

    def __init__ (self, input_size, learning_rate, initial_max_weight, initial_min_weight):
        self.weights = [randint (initial_min_weight, initial_max_weight) for i in range(input_size + 1)]
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.insure_bias_input = lambda input: input[:] if len(input) == self.input_size + 1 else input + [1]

    def activation (self, input):
        input = self.insure_bias_input(input)
        acc = 0
        for (x, w) in zip(input, self.weights):
            acc += x * w
        return acc

    def train (self, inputs, labels):
        loops_since_last_error = 0
        cases = [{'input': i + [1], 'label': l} for (i, l) in zip(inputs , labels)]
        age, i = 0, 0

        while loops_since_last_error < len(cases):
            error = cases[i]['label'] - self.predict(cases[i]['input'])

            if error != 0:
                loops_since_last_error = 0
                self.update_weights(cases[i]['input'], error)
            else:
                loops_since_last_error += 1
            
            age += 1
            i = age % len(cases)
            if age == 0: shuffle(cases)

    def update_weights(self, input, error):
        for (j, coordenate) in enumerate(input):
            self.weights[j] += self.learning_rate * error * coordenate

    def predict (self, input):
        return 1.0 if self.activation(input) >= 0.0 else 0.0


class Perceptron:
    def __init__(self, num_association_neurons, input_size, learning_rate, initial_max_weight, initial_min_weight):
        self.association_neurons = [PerceptronNeuron(input_size, learning_rate, initial_max_weight, initial_min_weight) for i in range (num_association_neurons)]
        self.response_neuron = PerceptronNeuron(input_size, learning_rate, initial_max_weight, initial_min_weight)

    def predict(self, input):
        self.association_output = [n.predict(input) for n in self.association_neurons]
        return self.response_neuron.predict(self.association_output)

    def train (self, inputs, labels):
        loops_since_last_error = 0
        cases = [{'input': i + [1], 'label': l} for (i, l) in zip(inputs , labels)]
        age, i = 0, 0

        while loops_since_last_error < len(cases):
            print("response:", self.response_neuron.weights)
            print("associacion 1:", self.association_neurons[0].weights)
            print("associantio 2:", self.association_neurons[1].weights)
            print("cases[i]['label']:", cases[i]['label'], "predict:", self.predict(cases[i]['input']))
            print("=========================================================================")
            error = cases[i]['label'] - self.predict(cases[i]['input'])

            if error != 0:
                loops_since_last_error = 0
                for (p, outputs) in zip(self.association_neurons + [self.response_neuron], self.association_output + [1]):
                    if (outputs == 1): 
                        p.update_weights(cases[i]['input'], error)
            else:
                loops_since_last_error += 1

            age += 1
            i = age % len(cases)
            if age == 0: shuffle(cases)



def test_train_and_predict(training_payload):
    possibilities = [[x, y] for x in range(2) for y in range(2)]
    sp = PerceptronNeuron(2, 0.5, 2, -2)

    sp.train(possibilities, training_payload)

    print("==========================")
    for p in possibilities: 
        print(p, "=> ", "verdadeiron" if sp.predict (p) else "falsidade")
    print("\npesos:", sp.weights)
    print("==========================")


# test_train_and_predict([0.0, 0.0, 0.0, 1.0])
# test_train_and_predict([0.0, 1.0, 1.0, 1.0])

with open("iris.csv", newline="\n") as csvfile:
    irirs_dataset = csv.reader(csvfile, delimiter = ",")
    dataset_input, dataset_labels = [], []

    for [sepal_length, sepal_width, petal_length, petal_width, spicie] in irirs_dataset:
        dataset_input.append(list(map(float, [sepal_length, sepal_width, petal_length, petal_width])))
        dataset_labels.append(1 if spicie == "Iris-setosa" else 0)
    dataset_input_size = len(dataset_input[0])
    dataset_size = len(dataset_input)

    trainig_dataset_input, trainig_dataset_labels = dataset_input[:int(dataset_size*0.8)], dataset_labels[:int(dataset_size*0.8)]
    test_dataset_input, test_dataset_labels = dataset_input[int(dataset_size*0.8):], dataset_labels[int(dataset_size*0.8):]

    sp = PerceptronNeuron(dataset_input_size, 0.1, 10, -10)
    sp.train(trainig_dataset_input, trainig_dataset_labels)

    correct_cases = 0
    for (input, label) in zip(test_dataset_input, test_dataset_labels):
        correct_cases += 1 if label == sp.predict(input) else 0

    print(correct_cases/len(test_dataset_input))
