from random import shuffle
from functools import reduce


def shuffle_lists(*lists):
    l = list(zip(*lists))
    shuffle(l)
    return zip(*l)

calc_average = lambda values: sum(values) / len(values)
calc_standard_deviation = lambda values, average_value: reduce(lambda acc, v: acc + abs(average_value-v), values, 0) / len(values)