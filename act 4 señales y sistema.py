#-------------------------------------------------------------------------------
# Name:        Implementación del Sistema de Modulación AM
# Purpose:
#
# Author:      moises cruz cruz
#
# Created:     27/07/2025
# Copyright:   (c) moise 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parámetros de la señal
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1, fs, endpoint=False)  # Vector de tiempo (1 segundo)

# Señal de mensaje (baja frecuencia)
fm = 5  # Frecuencia del mensaje (Hz)
Am = 1  # Amplitud del mensaje
mensaje = Am * np.sin(2 * np.pi * fm * t)

# Señal portadora (alta frecuencia)
fc = 50  # Frecuencia de la portadora (Hz)
Ac = 1  # Amplitud de la portadora
portadora = Ac * np.sin(2 * np.pi * fc * t)

# Modulación AM (DSB-SC: Double Sideband Suppressed Carrier)
senal_modulada = mensaje * portadora

# Visualización en el dominio del tiempo
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, mensaje, 'b')
plt.title("Señal de Mensaje (Dominio del Tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 2)
plt.plot(t, portadora, 'g')
plt.title("Señal Portadora (Dominio del Tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 3)
plt.plot(t, senal_modulada, 'r')
plt.title("Señal Modulada AM (Dominio del Tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# Análisis en frecuencia
N = len(senal_modulada)
yf = fft(senal_modulada)
xf = fftfreq(N, 1/fs)[:N//2]  # Solo frecuencias positivas

plt.figure(figsize=(10, 5))
plt.plot(xf, 2/N * np.abs(yf[0:N//2]))
plt.title("Espectro de Frecuencia de la Señal Modulada AM")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()
plt.show()

# Añadir ruido a la señal modulada
SNR_dB = 10  # Relación señal-ruido (dB)
potencia_senal = np.mean(senal_modulada**2)
potencia_ruido = potencia_senal / (10 ** (SNR_dB / 10))
ruido = np.random.normal(0, np.sqrt(potencia_ruido), len(senal_modulada))
senal_ruidosa = senal_modulada + ruido

# Visualización con ruido
plt.figure(figsize=(10, 5))
plt.plot(t, senal_ruidosa, 'm')
plt.title("Señal Modulada AM con Ruido (SNR = 10 dB)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

# Simular distorsión no lineal (compresión)
senal_distorsionada = np.tanh(senal_modulada * 2)  # Función no lineal

plt.figure(figsize=(10, 5))
plt.plot(t, senal_distorsionada, 'purple')
plt.title("Señal Modulada AM con Distorsión No Lineal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()