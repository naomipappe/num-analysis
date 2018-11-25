import eigen


matrix = [[1, 3, -2, 0, -2],
          [3, 4, -5, 1, -3],
          [-2, -5, 3, -2, 2],
          [0, 1, -2, 5, 3],
          [-2, -3, 2, 3, 4]]
lambda_max_mod = eigen.max_module_eigenvalue(matrix)
lambda_min_mod = eigen.min_module_eigenvalue(matrix)
lambda_min = eigen.min_eigenvalue(matrix)
lambdas = eigen.jacobi_turn_method(matrix)
print("Максимальное по модулю собственное число:", lambda_max_mod)
print("Минимальное по модулю собственное число:", lambda_min_mod)
print("Минимальное собственное число:", lambda_min)
print("Собственные значения матрицы:")
print(lambdas)
