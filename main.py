# Importación de las dependencias.
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Se leen los datos del archivo csv.
datos = pd.read_csv('altura_peso.csv')
alturas = datos['Altura'].values
pesos = datos['Peso'].values

# Utilización del Modelo en Keras.
modelo = Sequential([
    Input(shape=(1,)),
    Dense(1, activation='linear')
])
optimizador = SGD(learning_rate=0.0004)
modelo.compile(optimizer=optimizador, loss='mse')

# Se entrena el modelo.
historial = modelo.fit(alturas, pesos, epochs=10000, batch_size=len(alturas), verbose=0)

# Visualización de los datos, obtenemos y mostramos los parámetros w y b.
peso_neurona, sesgo = modelo.layers[0].get_weights()
print(f"Peso de la neurona: {peso_neurona[0][0]}, Sesgo: {sesgo[0]}")

# Grafica de la pérdida durante las épocas.
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'])
plt.title('MSE vs Épocas')
plt.xlabel('Épocas')
plt.ylabel('MSE')

# Gráfico de la recta de regresión de los datos.
plt.subplot(1, 2, 2)
plt.scatter(alturas, pesos, color='blue', label='Datos originales')
plt.plot(alturas, modelo.predict(alturas), color='red', label='Recta de regresión')
plt.title('Regresión Lineal con Keras')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# Predicción.
altura_a_predecir = 176
altura_np = np.array([[altura_a_predecir]])
peso_predicho = modelo.predict([altura_np])
print(f"El peso predicho para una altura de {altura_a_predecir} cm es {peso_predicho[0][0]:.2f} kg")
