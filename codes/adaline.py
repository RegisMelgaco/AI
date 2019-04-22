from functools import reduce
from random import shuffle, uniform


def predict(inputs, weights):
    return sum([w * i for (w, i) in zip(weights, inputs)])

def fit(training_payload, labels, learn_rate, max_eras=30000, debug=False):
    weights = gen_random_weights(len(training_payload[0])+1)

    era, hit_all = 0, False
    while(era < max_eras and not hit_all and learn_rate != 0):
        if debug:
            print("era:", era, "learn_rate:", learn_rate)
            print("weights", weights)
        hit_all = True
        for (inputs, label) in zip(training_payload, labels):
            error = label - predict(inputs, weights)
            if (error != 0):
                hit_all = False
                new_weights = [w + error*learn_rate*i for (w, i) in zip(weights, inputs+[1])]
                if new_weights == weights:
                    learn_rate = learn_rate / 10
                weights = new_weights
        era += 1

    return weights

def gen_random_weights(weights_num):
    return [uniform(-1, 1) for i in range(weights_num)]