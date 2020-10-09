import numpy as np

test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
flat_result = []
[rows, columns] = test.shape
for i in range(rows):
    for j in range(columns):
        flat_result.append(test[i][j])
print(flat_result)
