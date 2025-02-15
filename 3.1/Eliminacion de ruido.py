import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.fftpack as fft
import os


# Obtener la ruta de la carpeta donde está el script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta completa a la imagen
image_path = os.path.join(current_dir, "Ruido_E02.png")

image = Image.open(image_path).convert('L')    #Conver("L"), converts image into gray scale

#image=Image.open('/Users/Asus/Desktop/Corona/Programacion/Segunda entrega Instrumentos Opticos/Ruido_E02.png').convert('L')    #Conver("L"), converts image into gray scale

U0 = np.array(image, dtype=np.float64)             #float64 to use 64 bit numbers for mor precision 



def Propagation1 (U0,λ):

    Uz = np.fft.fftshift(fft.fft2(U0))     #Compute A(p,q,0) with FFT
    

    Uz = Uz*(1/(1j*λ*f1))
    return Uz


#1. Se tiene amplitud de entrada representada por la imagen
#2. La imagen se propaga hasta una lente de distancia focal f1 = 500mm y diametro de 100mm.
#3. Desde la lente, se propaga una distancia f1, interactua con una mascara que elimina el ruido aditivo.

# Pasos 1 a 3 descritos por la representacion de matrices ABCD en difraccion, como la transformada de fourier del campo de entrada multiplicado por 1/(i*lambda*f1)


#Ancho de imagen 768pixeles


λ= 6.5*10e-6            #longitud de onda en metros

M = 0.07                   #Magnificacion en el plano de salida.     Se elige esta magnificación para que cada pixel del plano de entrada corresponda a un pixel en el plano de salida

f1 = 0.5                #distancia focal f1 en metros
f2 = f1*M               #distancia focal f2 en metros, distancia configurada para que la imagen de salida tenga el tamaño del sensor de la camara

d1 = 0.1                #diametro de la lente 1 en m
d2 = 0.1                #diametro de la lente 2 en m

D = f2                  #distancia desde mascara hasta lente

P = d1                  #diametro de la pupila para modelar los diametros finitos de la lente 1 y 2

pixel = 3.45*10e-6       #Tamaño del pixel de salida (camara)
pixel0 = 130*10e-6       #Tamaño del pixel de entrada en metros


#Campo optico en el plano de la pupila y filtro de ruido

Uxy = Propagation1 (U0,λ)
Uxyabs = abs(Uxy)

#Uxy = np.log1p(np.abs(Uxy))

#Uxy = abs(Propagation1 (U0,λ))




"""
# Transformada inversa
Uz = np.fft.ifftshift(Uxymasked)  # Invertir el desplazamiento
image_reconstructed = np.fft.ifft2(Uz)  # Transformada inversa
image_reconstructed = np.abs(image_reconstructed)  # Tomar la magnitud real



Puntos
384
(408,322)
(426, 366)
(414,372)
"""




#MASCARAAAAAAAAAAAAAAAAAAAAAAAAAAA

# Create a mask to attenuate the bright spots in the spectrum
rows, cols = Uxyabs.shape
crow, ccol = rows // 2, cols // 2

# Initialize mask with ones
mask = np.ones((rows, cols), dtype=np.float64)

# Define positions of bright spots manually based on observation
radius = 10  # Radius around each bright spot to attenuate
bar_width = 0  # Width of the bars in pixels

# Coordinates of the bright spots to mask (adjusted manually after observing the spectrum)
bright_spots = [
    (323, 408), (2 * crow - 323, 2 * ccol - 408),
    (366, 426), (2 * crow - 366, 2 * ccol - 426),
    (372, 414), (2 * crow - 372, 2 * ccol - 414)
]

# Apply circular attenuation and bars around each bright spot
y, x = np.ogrid[:rows, :cols]
for cy, cx in bright_spots:
    # Circular attenuation
    distance = (x - cx) ** 2 + (y - cy) ** 2
    mask[distance <= radius ** 2] = 0  # Set mask to 0 in these regions

    # Horizontal bar attenuation
    mask[max(0, cy - bar_width):min(rows, cy + bar_width + 1), :] = 0

    # Vertical bar attenuation
    mask[:, max(0, cx - bar_width):min(cols, cx + bar_width + 1)] = 0

# Define the global circular mask to block outside a certain radius
global_radius = 384  # Radius of the allowed circular region
global_distance = (x - ccol) ** 2 + (y - crow) ** 2
mask[global_distance > global_radius ** 2] = 0  # Block everything outside the circle

# Apply mask to the Fourier-transformed image
f_masked = Uxy * mask

# Inverse Fourier Transform to reconstruct the image
f_ishifted = np.fft.ifftshift(f_masked)  # Shift back the frequencies
image_reconstructed = np.fft.ifft2(f_ishifted)  # Inverse FFT
image_reconstructed = np.abs(image_reconstructed)  # Get the real part

# Calcular la escala logarítmica del espectro
Uxy_log = np.log1p(np.abs(Uxy))

# Normalizar los valores a rango 0-255 (para guardar como imagen 8 bits)
Uxy_log_normalized = cv2.normalize(Uxy_log, None, 0, 255, cv2.NORM_MINMAX)

# Convertir a tipo uint8
Uxy_log_normalized = np.uint8(Uxy_log_normalized)

# Guardar la imagen en el directorio actual
output_path = os.path.join(current_dir, "Uxy spectre.png")
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
plt.title("Mask Applied to Spectrum")
plt.imshow(np.log1p(np.abs(f_masked)), cmap='gray', extent=[-3.84, 3.84, -3.84, 3.84])
plt.xlabel("kx (mm^-1)")
plt.ylabel("ky (mm^-1)")

# Normalizar espectro
i_masked = cv2.normalize(np.log1p(np.abs(f_masked)), None, 0, 255, cv2.NORM_MINMAX)
# Convertir a tipo uint8
i_masked_normalized = np.uint8(i_masked)

# Guardar el espectro con mascara
output_path_reconstructed = os.path.join(current_dir, "Masked_Spectre.png")
cv2.imwrite(output_path_reconstructed, i_masked_normalized)


# Reconstructed image
plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
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
output_path_reconstructed = os.path.join(current_dir, "Denoised_Image.png")
cv2.imwrite(output_path_reconstructed, image_reconstructed_normalized)

#-----------------------------------------------------------------------------------------------------------------

"""


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