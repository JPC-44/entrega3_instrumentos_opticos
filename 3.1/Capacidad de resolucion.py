import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.fftpack as fft
import scipy.ndimage
import os


# Obtener la ruta de la carpeta donde está el script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta completa a la imagen
image_path = os.path.join(current_dir, "DIEGO_2025_resolution_test_chart.png")
#image_path = os.path.join(current_dir, "USAF-1961 1px = 274nm.png")

image = Image.open(image_path).convert('L')    #Conver("L"), converts image into gray scale

U0 = np.array(image, dtype=np.float64)             #float64 to use 64 bit numbers for mor precision 


def Propagation1(U0, λ, NA, pixel, M, N):
    """
    Aplica la Transformada de Fourier a una imagen y filtra las frecuencias espaciales
    superiores a NA/lambda mediante una máscara en el dominio de Fourier.

    Parámetros:
    - image: Imagen de entrada en escala de grises (numpy array).
    - wavelength: Longitud de onda en metros (lambda).
    - pixel_size: Tamaño del píxel en metros.
    - NA: Apertura numérica del sistema óptico.

    Retorna:
    - image_filtered: Imagen filtrada después de aplicar la pupila en el dominio de Fourier.
    - fourier_mask: Máscara de pupila utilizada en el plano de Fourier.
    """

    # Calcular la frecuencia máxima permitida por el sistema óptico
    f_max = (2*NA / λ)               #Se multiplica *2 ya que las frecuencias de np.fft.fftfreq estan normalizadas por lo que se crearia una mascara de la mitad del diametro si no se multiplica

    # Calcular el tamaño del píxel en el dominio de Fourier
    delta_f_x = 1 / (N * pixel)
    delta_f_y = 1 / (N * pixel)

    # Crear el espacio de frecuencias
    fx = np.fft.fftfreq(N,pixel0)
    fy = np.fft.fftfreq(N, pixel0)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))

    # Crear la máscara de pupila en el dominio de Fourier
    fourier_mask = np.sqrt(FX**2 + FY**2) <= f_max

    # Aplicar la Transformada de Fourier a la imagen
    image_fft = np.fft.fftshift(np.fft.fft2(image))

    # Aplicar la máscara de pupila
    filtered_fft = image_fft * fourier_mask


    return filtered_fft, fourier_mask

# Ejemplo de uso (suponiendo que 'image' ya está cargada como una matriz numpy):
# image_filtered, mask = apply_fourier_pupil_mask(image, 533e-9, 2.74e-6, 0.25)


#1. Se tiene amplitud de entrada representada por la imagen
#2. La imagen se propaga hasta el objetivo de microscopio con f1 = 20mm
#3. Desde el objetivo, se propaga una distancia f1, interactua con la pupila que limita las frecuencias del sistema segun la apertura numerica


#Ancho de imagen 2848pixeles


λ= 533*10e-9            #longitud de onda en metros

M = 10                   #Magnificacion en el plano de salida.     Se elige esta magnificación para que cada pixel del plano de entrada corresponda a un pixel en el plano de salida

#M = f2/f1
f2 = 0.2                #distancia focal f2 en metros, distancia configurada para que la imagen de salida tenga el tamaño del sensor de la camara
f1 = f2/M               #distancia focal f1 en metros

D = f2                  #distancia desde mascara hasta lente de tubo 

N = 2848                #Cantidad de pixeles en la camara

pixel = 2.74*10e-6          #Tamaño del pixel de salida (camara)
pixel0 = pixel/M            #Tamaño del pixel de entrada en metros
pixelf = 1/N*pixel0         #Tamaño del pixel en el espacio de frecuencia


FOV = N*pixel0             #Campo de vision del plano de entrada en metros

NA=0.25                     #Numerical Aperture

f_max=2*NA/λ                     #Maxima frecuencia que admite el sistema


#Campo optico en el plano de la pupila

Uxy,mask = Propagation1 (U0,λ, NA, pixel, M, N)

Uxy_reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(Uxy)))

Uxy_reconstructed_norm = cv2.normalize(Uxy_reconstructed, None, 0, 255, cv2.NORM_MINMAX)

# Convertir a tipo uint8 para visualización correcta
Uxy_reconstructed = np.uint8(Uxy_reconstructed_norm)

# Mostrar el espectro de Fourier (Uxy) y la máscara de pupila (mask)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Espectro de Fourier del campo óptico en la pupila
ax[0].imshow(np.log1p(np.abs(U0)), cmap='gray', extent=[-FOV/20,FOV/20, -FOV/20, FOV/20])
ax[0].set_title("Imagen de entrada (1 px = 274 nm )")
ax[0].set_xlabel("x (m)")
ax[0].set_ylabel("y (m)")


"""
# Espectro de Fourier del campo óptico en la pupila
ax[1].imshow(np.log1p(np.abs(Uxy)), cmap='gray', extent=[-f_max, f_max, -f_max, f_max])
ax[1].set_title("Espectro de Fourier con Pupila")
ax[1].set_xlabel("Frecuencia espacial (1/m)")
ax[1].set_ylabel("Frecuencia espacial (1/m)")
"""

# Imagen reconstruida a partir del espectro de Fourier filtrado
ax[1].imshow(Uxy_reconstructed, cmap='gray', extent=[-FOV/2,FOV/2, -FOV/2, FOV/2])
ax[1].set_title("Imagen magnificada 10X (1 px = 2.74 um )")
ax[1].set_xlabel("x (m)")
ax[1].set_ylabel("y (m)")



# Inverse Fourier Transform to reconstruct the image
f_ishifted = np.fft.ifftshift(Uxy)  # Shift back the frequencies
image_reconstructed = np.fft.ifft2(f_ishifted)  # Inverse FFT
image_reconstructed = np.abs(image_reconstructed)  # Get the real part




plt.tight_layout()
plt.show()


import numpy as np

# Convertir la imagen PIL a un array NumPy si aún no lo es
if not isinstance(image_reconstructed, np.ndarray):
    image_reconstructed = np.array(image_reconstructed, dtype=np.uint8)
# Aplicar un suavizado a la señal para visualizarla como una curva
image_reconstructed =  np.abs(image_reconstructed-255)
y_center = image_reconstructed.shape[0] // 2  # Tomar la fila central
intensity_profile =  image_reconstructed[y_center, :]  # Extraer la intensidad a lo largo del eje X
smoothed_intensity = scipy.ndimage.gaussian_filter1d(intensity_profile, sigma=2)

# Graficar la intensidad suavizada en función de la posición en X
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(smoothed_intensity)), smoothed_intensity, color='black', linewidth=2)
plt.xlabel("Posición en X (píxeles)")
plt.ylabel("Intensidad (0 = blanco, 255 = negro)")
plt.title("Perfil de Intensidad a lo Largo del Eje X (Suavizado)")
plt.grid(True)
plt.show()



"""
# Inverse Fourier Transform to reconstruct the image
f_ishifted = np.fft.ifftshift(Uxy)  # Shift back the frequencies
image_reconstructed = np.fft.ifft2(f_ishifted)  # Inverse FFT
image_reconstructed = np.abs(image_reconstructed)  # Get the real part

# Calcular la escala logarítmica del espectro
Uxy_log = np.log1p(np.abs(Uxy))

# Normalizar los valores a rango 0-255 (para guardar como imagen 8 bits)
Uxy_log_normalized = cv2.normalize(Uxy_log, None, 0, 255, cv2.NORM_MINMAX)

# Convertir a tipo uint8
Uxy_log_normalized = np.uint8(Uxy_log_normalized)

# Guardar la imagen en el directorio actual
output_path = os.path.join(current_dir, "Uxy spectre with pupil.png")
cv2.imwrite(output_path, Uxy_log_normalized)

# Display the results
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(U0, cmap='gray', extent=[-50, 50, -50, 50])
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

# Mask applied to the spectrum
plt.subplot(1, 3, 2)
plt.title("Spectrum with pupil")
plt.imshow(np.log1p(np.abs(Uxy)), cmap='gray', extent=[-3.84, 3.84, -3.84, 3.84])
plt.xlabel("kx (mm^-1)")
plt.ylabel("ky (mm^-1)")

# Normalizar espectro
i_masked = cv2.normalize(np.log1p(np.abs(Uxy)), None, 0, 255, cv2.NORM_MINMAX)
# Convertir a tipo uint8
i_masked_normalized = np.uint8(i_masked)

# Guardar el espectro con mascara
output_path_reconstructed = os.path.join(current_dir, "Spectre with pupil.png")
cv2.imwrite(output_path_reconstructed, i_masked_normalized)


# Reconstructed image
plt.subplot(1, 3, 3)
plt.title("Magnified image")
plt.imshow(image_reconstructed, cmap='gray', extent=[-3.5, 3.5, -3.5, 3.5])
plt.xlabel("u (mm)")
plt.ylabel("v (mm)")

plt.tight_layout()
plt.show()

# Normalizar la imagen reconstruida al rango 0-255
image_reconstructed_normalized = cv2.normalize(image_reconstructed, None, 0, 255, cv2.NORM_MINMAX)
# Convertir a tipo uint8
image_reconstructed_normalized = np.uint8(image_reconstructed_normalized)

# Guardar la imagen reconstruida
output_path_reconstructed = os.path.join(current_dir, "Magnified image.png")
cv2.imwrite(output_path_reconstructed, image_reconstructed_normalized)

#-----------------------------------------------------------------------------------------------------------------




# Mostrar la imagen original y la transformada de Fourier
plt.figure(figsize=(12, 6))

# Imagen original
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(U0, cmap='gray')
plt.axis('off')

# Transformada de Fourier
plt.subplot(1, 2, 2)
plt.title("Transformada de Fourier 2D")
plt.imshow(Uxy, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
"""