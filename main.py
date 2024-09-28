import numpy as np
import matplotlib.pyplot as plt

# Passo 1: Definindo os vetores e a matriz de regressão
# 1a. Vetor de saídas desejadas Y (5x1)
Y = np.array([0.2, 0.3, 0.45, 0.7, 0.8])

# 1b. Vetor de entradas aplicadas X (5x1)
X = np.array([0, 0.1, 0.2, 0.3, 0.4])

# 1c. Matriz de regressão XX (5x2)
XX = np.column_stack((X, np.ones(X.shape[0])))

# Passo 2: Resolva o sistema Y = XX.COEF usando o método dos mínimos quadrados
# Usamos o operador "@" para a multiplicação matricial e np.linalg.lstsq para resolver mínimos quadrados
COEF, _, _, _ = np.linalg.lstsq(XX, Y, rcond=None)

# Coeficientes a e b
a, b = COEF
print(f"Coeficiente a: {a}")
print(f"Coeficiente b: {b}")

# Passo 3: Usando o vetor de coeficientes para calcular as saídas estimadas
Y_est = XX @ COEF  # Multiplicação matricial para obter os valores estimados

# Passo 4: Visualização
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Dados observados (Y)')
plt.plot(X, Y_est, color='red', label='Reta ajustada (XX.COEF)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste de mínimos quadrados: Y vs X')
plt.legend()
plt.grid(True)
plt.show()
