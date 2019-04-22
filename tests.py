from codes.uni_layer_perceptron import activation, predict, calc_errors, fit

# act = activation([2,3], [5,7,11])

# print("act:", act == 42)

# w1 = [[5,7,-32],[5,7,11]]
# w2 = [[5,7,11],[5,7,-32]]

# pre1 = predict([2,3], w1)

# pre2 = predict([3,2], w2)

# print("predict:", pre1 == 1 and pre2 == 0)

# ce1 = calc_errors([7,7], w1, pre1, 2)

# ce2 = calc_errors([2,2], w2, pre1, 2)

# print(ce1, ce2)
# print("calc_errors:", ce2 == [1,0] and ce1 == [-1,-1])

i = [[i, j] for i in range(2) for j in range(2)]
w = fit(i, [0,1,1,1], 2, 0.1, 10000)
p = [predict(j, w) for j in i]

print(p)