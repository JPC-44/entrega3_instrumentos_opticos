import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.fftpack as fft
import os

# Importar máscara para Z=0


image = Image.open(r"C:/Users/Usuario/Desktop/GitHub/entrega3_instrumentos_opticos/Hologram.tiff").convert('L')  # Convertir a escala de grises




Campo = np.array(image)
U0 = Campo
λ = 632.80E-9  # Longitud de onda en metros
pixel = 3.45E-6  # Tamaño de píxel en metros



# Crear carpeta para guardar las imágenes
output_folder = "imagenes_propagadas"
os.makedirs(output_folder, exist_ok=True)

# Generar imágenes y guardarlas

UZ_magnitude = (np.abs(U0))**2

U0= np.abs(U0)**2

output_path = os.path.join(output_folder, f"IntensityCalculated.png")
plt.imsave(output_path, UZ_magnitude, cmap='gray')
print(f"Imagen guardada: {output_path}")

print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")




def AngularSpectrum(U0, Z, λ, pixel):
    M, N = U0.shape
    fx = np.fft.fftshift(np.fft.fftfreq(N, pixel))
    fy = np.fft.fftshift(np.fft.fftfreq(M, pixel))
    fx, fy = np.meshgrid(fx, fy)

    K = 2 * np.pi / λ
    H = np.exp(-1j * Z * np.sqrt(K**2 - (2 * np.pi * fx)**2 - (2 * np.pi * fy)**2))

    mask = np.sqrt(fx**2 + fy**2) <= 1 / λ
    H *= mask

    A0 = np.fft.fftshift(fft.fft2(U0))
    Az = A0 * H
    Uz = fft.ifft2(np.fft.fftshift(Az))

    return Uz

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Por favor, ingrese un número válido.")

def get_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Por favor, ingrese un número entero válido.")

# Parámetros ajustables
start_Z = get_float_input("Ingrese el valor inicial de Z (en metros): ")
step_Z = get_float_input("Ingrese el paso entre valores de Z (en metros): ")
num_images = get_int_input("Ingrese la cantidad de imágenes a generar: ")

# Crear carpeta para guardar las imágenes
output_folder = "imagenes_propagadas"
os.makedirs(output_folder, exist_ok=True)

# Generar imágenes y guardarlas
for i in range(num_images):
    Z = start_Z + i * step_Z
    UZ = AngularSpectrum(U0, Z, λ, pixel)
    UZ_magnitude = np.abs(UZ)**2

    output_path = os.path.join(output_folder, f"propagated_Z_{Z:.5f}m.png")
    plt.imsave(output_path, UZ_magnitude, cmap='gray')
    print(f"Imagen guardada: {output_path}")

print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")