from random import randint
from functools import reduce



class SimplePerceptron:

    def __init__ (self, input_size, learning_rate, initial_max_weight, initial_min_weight):
        self.weights = [randint (initial_min_weight, initial_max_weight) for i in range(input_size + 1)]
        self.input_size = input_size
        self.learning_rate = learning_rate

    def limiar_function (self, input):
        acc = 0
        for (x, w) in zip(input, self.weights):
            acc += x * w
        return acc

    def train (self, inputs, labels):
        loops_since_last_error = 0
        i = 0

        while True:
            error = labels[i].numerator - self.predict(inputs[i]).numerator

            if error != 0:
                loops_since_last_error = 0
                for (j, coordenate) in enumerate(inputs[i]):
                    self.weights[j] += self.learning_rate * error * coordenate
            else:
                loops_since_last_error += 1
                if loops_since_last_error > len(labels):
                    break
            
            if i < len(labels) - 1:
                i += 1
            else:
                i = 0

    def predict (self, input):
        return self.limiar_function(input) >= 0


sp = SimplePerceptron(2, 0.5, 2, -2)
# sp.weights = [1, 1, 1.5]

possibilities = [[x, y] for x in range(2) for y in range(2)]
expanded_p = [p + [-1] for p in possibilities]

sp.train(expanded_p, [False, False, False, True])

for p in expanded_p: 
    print(p, "=> ", sp.predict (p))
