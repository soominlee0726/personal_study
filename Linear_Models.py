from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)         # w vector is coefficient
print(reg.intercept_)    # constant b is intercept

example1 = linear_model.LinearRegression()
example1.fit([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], [5, 10, 15, 20])

''' calculation
w2 + 2*w3 + b = 5
w1 + 2*w2 + 3*w3 +b = 10
2*w1 + 3*w2 + 4*w3 +b = 15
3*w1 + 4*w2 + 5*w3 +b = 20
'''

print(example1.coef_)
print(example1.intercept_)


reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(f'when alpha is 0.5, coefficient is {reg.coef_}')
print(f'when alpha is 0.5, intercept is {reg.intercept_}')

example2 = linear_model.Ridge(alpha = 10)
example2.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(f'when alpha is 10, coefficient is {example2.coef_}')
print(f'when alpha is 10, intercept is {example2.intercept_}')