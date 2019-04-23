# from codes.adaline import predict, fit
# from random import uniform

# fun1 = lambda i: i[0]*2
# fun2 = lambda i: i[0]*3 + i[1]*-2 + i[2]*0.5

# training_payload = [[uniform(-10, 10) for j in range(3)] for i in range(40)]
# labels = [fun2(i) for i in training_payload]
# # labels = [fun1(i) for i in training_payload]

# weights = fit(training_payload, labels, 0.0001, 3000000)

# test = [predict(i, weights) for i in training_payload]
# for (t, l) in zip(test, labels):
#     print("t:", t, "l:", l)

# print(weights)


from codes.common import shuffle_lists

a, b, c = shuffle_lists([1,2,3], [4,5,6], [7,8,9])

print(a,b,c)