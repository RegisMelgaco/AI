from random import randint, shuffle
from functools import reduce


class PerceptronNeuron:

    def __init__ (self, input_size, learning_rate, initial_max_weight, initial_min_weight, max_age = 10000):
        self.weights = self.gen_weights(input_size, initial_min_weight, initial_max_weight)
        self.bias = self.gen_bias(initial_min_weight, initial_max_weight)
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.age, self.max_age = 0, max_age

    def gen_weights(self, input_size, min_weight, max_weight):
        weights = []
        for i in range(1, input_size+1):
            weights.append([randint (min_weight, max_weight) for j in range(i)]) 
        return weights

    def gen_bias(self, min_weight, max_weight):
        return randint (min_weight, max_weight)

    def activation (self, input):
        acc = self.bias
        for (row, w_row) in enumerate(self.weights):
            for (col, weight) in enumerate(w_row):
                acc += weight * input[row] if row == col else weight * input[row] * input[col]
        return acc

    def update_weights(self, input, error):
        for row in range(len(self.weights)):
            for col in range(len(self.weights[row])):
                if row == col:
                    self.weights[row][col] += input[row] * error * self.learning_rate
                else:
                    self.weights[row][col] += input[row] * input[col] * error * self.learning_rate

    def update_bias(self, error):
        self.bias += self.learning_rate * error

    def fit (self, inputs, labels):
        loops_since_last_error = 0
        cases = [{'input': i + [1], 'label': l} for (i, l) in zip(inputs , labels)]
        self.age, i = 0, 0

        while loops_since_last_error < len(cases) and self.max_age >= self.age:
            error = cases[i]['label'] - self.predict(cases[i]['input'])

            if error != 0:
                loops_since_last_error = 0
                self.update_weights(cases[i]['input'], error)
                self.update_bias(error)
            else:
                loops_since_last_error += 1
            
            self.age += 1
            i = self.age % len(cases)
            if self.age == 0: shuffle(cases)

    def predict (self, input):
        return 1.0 if self.activation(input) >= 0.0 else 0.0
