import solver
matrix = [1, 3, -2, 0, 2,
          3, 4, -5, 1, -3,
          -2, -5, 3, -2, 2,
          0, 1, -2, 5, 3,
          -2, -3, 2, 3, 4]

test = [2, 1, 4,
        1, 1, 3,
        4, 3, 14]
b = [0.5, 5.4, 5.0, 7.5, 3.3]
solver.square_root_method(matrix, b)
