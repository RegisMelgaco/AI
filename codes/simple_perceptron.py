from random import randint, shuffle
from functools import reduce


class PerceptronNeuron:

    def __init__ (self, input_size, learning_rate, initial_max_weight, initial_min_weight, max_age = 10000):
        self.weights = [randint (initial_min_weight, initial_max_weight) for i in range(input_size + 1)]
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.age, self.max_age = 0, max_age

    def insure_bias_input(self, input):
        return input[:] if len(input) == self.input_size + 1 else input + [1]

    def activation (self, input):
        input = self.insure_bias_input(input)
        acc = 0
        for (x, w) in zip(input, self.weights):
            acc += x * w
        return acc

    def fit (self, inputs, labels):
        loops_since_last_error = 0
        cases = [{'input': i + [1], 'label': l} for (i, l) in zip(inputs , labels)]
        self.age, i = 0, 0

        while loops_since_last_error < len(cases) and self.max_age >= self.age:
            error = cases[i]['label'] - self.predict(cases[i]['input'])

            if error != 0:
                loops_since_last_error = 0
                self.update_weights(cases[i]['input'], error)
            else:
                loops_since_last_error += 1
            
            self.age += 1
            i = self.age % len(cases)
            if self.age == 0: shuffle(cases)

    def update_weights(self, input, error):
        for (j, coordenate) in enumerate(input):
            self.weights[j] += self.learning_rate * error * coordenate

    def predict (self, input):
        return 1.0 if self.activation(input) >= 0.0 else 0.0
