from random import randint, shuffle
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
        cases = [{'input': i, 'label': l} for (i, l) in zip(inputs, labels)]
        i = 0

        while True:
            shuffle(cases)
            error = cases[i]['label'] - self.predict(cases[i]['input'])

            if error != 0:
                loops_since_last_error = 0
                for (j, coordenate) in enumerate(cases[i]['input']):
                    self.weights[j] += self.learning_rate * error * coordenate
            else:
                loops_since_last_error += 1
                if loops_since_last_error > len(cases):
                    break
            
            if i < len(cases) - 1:
                i += 1
            else:
                i = 0

    def predict (self, input):
        return self.limiar_function(input) >= 0


sp = SimplePerceptron(2, 0.5, 2, -2)
# sp.weights = [1, 1, 1.5] # pesos para a AND

possibilities = [[x, y] for x in range(2) for y in range(2)]
expanded_p = [p + [-1] for p in possibilities]


print("==========================")

sp.train(expanded_p, [False, False, False, True])

for (ep, p) in zip(expanded_p, possibilities): 
    print(p, "=> ", sp.predict (ep))
    
print("==========================")
    
sp.train(expanded_p, [False, True, True, True])

for (ep, p) in zip(expanded_p, possibilities): 
    print(p, "=> ", sp.predict (ep))
    
print("==========================")
    
sp.train(expanded_p, [False, True, False, True])

for (ep, p) in zip(expanded_p, possibilities): 
    print(p, "=> ", sp.predict (ep))
    
print("==========================")
