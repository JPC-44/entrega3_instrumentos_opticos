import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Imagen en escala de grises
ruta_imagen = r"C:/Users/Usuario/Desktop/Ruido_E02.png"
image = Image.open(ruta_imagen).convert('L')
noisy_image = np.array(image)


def filterInAction():

    # FFT de la imagen
    fourier_transform = np.fft.fftshift(np.fft.fft2(noisy_image))
    
    filter = np.ones_like(fourier_transform)

    frequencies = [(324, 409), (366, 425),(373, 415)]  

    radio =  5  
    for (u, v) in frequencies:
        filter[u-radio:u+radio, v-radio:v+radio] = 0  
        filter[-u-radio:-u+radio, -v-radio:-v+radio] = 0  

    # Filtro para las los armónicos que introducen ruido en la imagen
    filtered_fourier = fourier_transform * filter

    # Imagen con ruido filtrada
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fourier)))

    # Plots para la comparación de la imagen y su espectro con y sin ruido
    plt.figure(figsize=(12, 6))

    # Noisy image
    plt.subplot(2, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(noisy_image, cmap='gray')

    # Espectro en log10 de la imagen con ruido
    plt.subplot(2, 2, 2)
    plt.title("Espectro Original")
    plt.imshow(np.log10(1 + np.abs(fourier_transform)), cmap='gray')


    # Espectro de la imagen con el filtro
    plt.subplot(2, 2, 3)
    plt.title("Imagen Filtrada")
    plt.imshow(np.log10(1+np.abs(filtered_fourier)), cmap='gray')

    # Imagen filtrada
    plt.subplot(2, 2, 4)
    plt.title("Imagen Filtrada")
    plt.imshow(filtered_image, cmap='gray')

    plt.show()


# el criterio para diseñar estos filtros son funciones RECT que están posicionadas en un delta de dirac desplazado en un armónico 
filterInAction()


