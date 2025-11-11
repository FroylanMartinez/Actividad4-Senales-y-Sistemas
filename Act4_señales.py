import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os # Para crear carpeta

print("Iniciando simulación AM...")

# --- 0. Configuración para guardar gráficas ---
output_folder = "Graficas_AM"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 1. Definición de Parámetros y Señales de Entrada ---
fs = 20000  # Frecuencia de muestreo (Hz)
T = 1.0     # Duración de la señal (segundos)
t = np.linspace(0, T, int(T * fs), endpoint=False) # Vector de tiempo

# Señal del mensaje (Información)
fm = 50     # Frecuencia del mensaje (50 Hz)
Am = 1.0    # Amplitud del mensaje
m_t = Am * np.cos(2 * np.pi * fm * t) # Señal del mensaje (moduladora)

# Señal Portadora (Carrier)
fc = 1000   # Frecuencia de la portadora (1000 Hz)
Ac = 1.0    # Amplitud de la portadora
c_t = Ac * np.cos(2 * np.pi * fc * t) # Señal portadora

# Índice de modulación
mu = 0.75  # (Submodulación, mu < 1)

# --- 2. Implementación de la Modulación AM ---
# Ecuación: s(t) = Ac * (1 + mu * m(t)/Am) * c(t)
# Asumiendo Am=1 para normalizar, m_norm = cos(2*pi*fm*t)
m_t_norm = np.cos(2 * np.pi * fm * t)
am_t = Ac * (1 + mu * m_t_norm) * c_t # Señal Modulada AM

# --- 3. Creación de Gráficas ---

# Duración para graficar en tiempo (para que se vean las ondas)
plot_duration = 0.1 # 100 ms
n_samples_plot = int(plot_duration * fs)

# GRÁFICA 1: Señales de Entrada (Mensaje y Portadora)
plt.figure(figsize=(14, 6))
plt.plot(t[:n_samples_plot], m_t[:n_samples_plot], label='Mensaje $m(t)$ (fm=50 Hz)')
plt.plot(t[:n_samples_plot], c_t[:n_samples_plot], 'k', label='Portadora $c(t)$ (fc=1000 Hz)', alpha=0.7)
plt.title('Gráfica 1: Señales de Entrada (Tiempo)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_folder, "plot_1_señales_entrada.png"))
print("Gráfica 1 guardada.")

# GRÁFICA 2: Señal Modulada AM (Tiempo)
plt.figure(figsize=(14, 6))
plt.plot(t[:n_samples_plot], am_t[:n_samples_plot], 'r', label='Señal Modulada $s(t)$')
# Dibujamos la envolvente
envolvente_sup = Ac * (1 + mu * m_t_norm[:n_samples_plot])
envolvente_inf = -Ac * (1 + mu * m_t_norm[:n_samples_plot])
plt.plot(t[:n_samples_plot], envolvente_sup, 'g--', label='Envolvente (Información)')
plt.plot(t[:n_samples_plot], envolvente_inf, 'g--')
plt.title('Gráfica 2: Señal Modulada AM (Tiempo)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_folder, "plot_2_am_tiempo.png"))
print("Gráfica 2 guardada.")


# --- 4. Análisis de Frecuencia (FFT) ---

# Función auxiliar para graficar FFT
def plot_fft(signal, fs, title, filename):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)
    
    # Graficamos la mitad positiva del espectro
    n_positive = N // 2
    yf_positive = 2.0/N * np.abs(yf[:n_positive])
    xf_positive = xf[:n_positive]
    
    plt.figure(figsize=(12, 6))
    plt.plot(xf_positive, yf_positive)
    plt.title(title)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    # Zoom en la parte interesante (alrededor de fc)
    plt.xlim(fc - 4*fm, fc + 4*fm) 
    plt.ylim(0, Ac + 0.2) # Ajustar límite Y
    plt.savefig(os.path.join(output_folder, filename))
    print(f"{filename} guardada.")

# GRÁFICA 3: FFT de la Señal Pura
plot_fft(am_t, fs, "Gráfica 3: Espectro de Frecuencia (FFT) - Señal AM Pura", "plot_3_am_frecuencia_pura.png")

# --- 5. Análisis de Ruido y Distorsión ---

# 5a. Introducir Ruido (AWGN)
noise_power = 0.8
noise = noise_power * np.random.normal(0, 1, size=t.shape)
am_t_ruido = am_t + noise

# GRÁFICA 4: Señal con Ruido (Tiempo)
plt.figure(figsize=(14, 6))
plt.plot(t[:n_samples_plot], am_t_ruido[:n_samples_plot], 'r', label='Señal con Ruido')
plt.title('Gráfica 4: Señal AM con Ruido (Tiempo)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_folder, "plot_4_am_ruido_tiempo.png"))
print("Gráfica 4 guardada.")

# GRÁFICA 5: FFT de la Señal con Ruido
plot_fft(am_t_ruido, fs, "Gráfica 5: Espectro de Frecuencia (FFT) - Señal AM con Ruido", "plot_5_am_frecuencia_ruido.png")

# 5b. Simulación de Distorsión (Clipping)
# El amplificador satura en 1.5 y -1.5
am_t_distorsion = np.clip(am_t, -1.5, 1.5)

# GRÁFICA 6: FFT de la Señal con Distorsión
plot_fft(am_t_distorsion, fs, "Gráfica 6: Espectro de Frecuencia (FFT) - Señal AM con Distorsión", "plot_6_am_frecuencia_distorsion.png")

print("\n--- Simulación Completa ---")
print(f"Todas las gráficas han sido guardadas en la carpeta: '{output_folder}'")
# plt.show() # Opcional: Descomenta si quieres que se muestren al final
