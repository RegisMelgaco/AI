from random import randint, shuffle
from functools import reduce



class SimplePerceptron:

    def __init__ (self, input_size, learning_rate, initial_max_weight, initial_min_weight):
        self.weights = [randint (initial_min_weight, initial_max_weight) for i in range(input_size + 1)]
        self.input_size = input_size
        self.learning_rate = learning_rate

    def _limiar_function (self, input):
        acc = 0
        for (x, w) in zip(input, self.weights):
            acc += x * w
        return acc

    def train (self, inputs, labels):
        loops_since_last_error = 0
        cases = [{'input': i + [1], 'label': l} for (i, l) in zip(inputs , labels)]
        age, i = 0, 0

        while loops_since_last_error <= len(cases):
            error = cases[i]['label'] - self._predict(cases[i]['input'])

            if error != 0:
                loops_since_last_error = 0
                for (j, coordenate) in enumerate(cases[i]['input']):
                    self.weights[j] += self.learning_rate * error * coordenate
            else:
                loops_since_last_error += 1
            
            age += 1
            i = age % len(cases)
            if age == 0: shuffle(cases)

    def _predict (self, input):
        return self._limiar_function(input) >= 0

    def predict (self, input): 
        return self._predict(input + [1])



def test_train_and_predict(training_payload):
    possibilities = [[x, y] for x in range(2) for y in range(2)]
    sp = SimplePerceptron(2, 0.5, 2, -2)

    sp.train(possibilities, training_payload)

    print("==========================")
    for p in possibilities: 
        print(p, "=> ", sp.predict (p))
    print("==========================")


test_train_and_predict([False, False, False, True])
test_train_and_predict([False, True, True, True])
test_train_and_predict([False, True, False, True])
