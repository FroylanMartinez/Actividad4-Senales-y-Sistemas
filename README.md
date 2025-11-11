# Actividad 4: Simulación de Modulación AM

Este proyecto es una simulación de un sistema de Modulación en Amplitud (AM) desarrollada para la asignatura de Señales y Sistemas.

## Propósito del Código

El script de Python (`Act4_señales.py`) realiza las siguientes tareas:

* Genera una señal de mensaje (baja frecuencia) y una señal portadora (alta frecuencia).
* Implementa la ecuación de modulación AM estándar.
* Grafica las señales en el dominio del tiempo (mensaje, portadora y señal modulada).
* Analiza la señal modulada en el dominio de la frecuencia usando la Transformada Rápida de Fourier (FFT).
* Simula el impacto del ruido aditivo (AWGN) y la distorsión (clipping) en la señal AM.
* Guarda todas las gráficas generadas en una carpeta local.

## Librerías Utilizadas

* **Numpy:** Para los cálculos numéricos y la creación de señales.
* **Matplotlib:** Para la generación de todas las gráficas.
* **Scipy:** Para el uso de la función FFT.
