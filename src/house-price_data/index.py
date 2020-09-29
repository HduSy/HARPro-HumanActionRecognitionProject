import csv
import numpy as np


def readData():
    X = []
    y = []
    with open('Housing.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        # Read X and y
        for line in rdr:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
            X.append(xline)
            y.append(float(line[-1]))
    return (X, y)


X0, y0 = readData()
print('X0:', X0)  # (546,12)
print('y0:', y0)  # (546,)
# Convert all but the last 10 rows of the raw data to numpy arrays
d = len(X0) - 10
X = np.array(X0[:d])  # (536,12)
print(X.shape)
y = np.transpose(np.array([y0[:d]]))
print('y.shape:', y.shape)  # (536,1)
# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt, X)
print('XtX.shape:', XtX.shape)  # (12,536)*(536,12)=>(12,12)
Xty = np.dot(Xt, y)  # (12,536)*(536,1)=>(12,1)
beta = np.linalg.solve(XtX, Xty)  # (12,12)*(12,1)=>(12,1)
# beta = np.linalg.solve(np.linalg.inv(X),y)
print(beta)  # (12,1)

# Make predictions for the last 10 rows in the data set
for data, actual in zip(X0[d:], y0[d:]):
    x = np.array([data])
    prediction = np.dot(x, beta)  # (1,12)*(12,1)=>(1,1)
    print('prediction = ' + str(prediction[0, 0]) + ' actual = ' + str(actual))

a = [1.0, 5850.0, 3.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
b = [[-4.14106096e+03],
     [3.55197583e+00],
     [1.66328263e+03],
     [1.45465644e+04],
     [6.77755381e+03],
     [6.58750520e+03],
     [4.44683380e+03],
     [5.60834856e+03],
     [1.27979572e+04],
     [1.24091640e+04],
     [4.19931185e+03],
     [9.42215457e+03]]
print(np.dot(a, b))
