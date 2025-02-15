import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

def filtro_rectangular(tamaño=(768, 768), ancho=100, alto=100):
    h, w = tamaño
    filtro = np.zeros((h, w))
    
    x_inicio = w//2 - ancho//2
    x_fin = w//2 + ancho//2
    y_inicio = h//2 - alto//2
    y_fin = h//2 + alto//2
    

    filtro[y_inicio:y_fin, x_inicio:x_fin] = 1
    
    return filtro

ruta_imagen = r"C:/Users/Usuario/Desktop/Ruido_E02.png"
Noise = Image.open(ruta_imagen).convert('L')
NoiseArray = np.array(Noise)

ImageFrequencies = np.fft.fftshift(np.fft.fft2(NoiseArray))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.25)

ancho_init = 100
alto_init = 100
filtro = filtro_rectangular(tamaño=NoiseArray.shape, ancho=ancho_init, alto=alto_init)
filtrado = filtro * ImageFrequencies
ImageSpectrum = np.abs(filtrado)
ImageFiltered = np.abs(np.fft.ifft2(np.fft.fftshift(filtrado)))

ax[0].imshow(NoiseArray, cmap='gray')
ax[0].set_title("Imagen Original")

im_spectrum = ax[1].imshow(np.log10(1 + ImageSpectrum), cmap='gray')
ax[1].set_title(f"Espectro Filtrado (Ancho={ancho_init}, Alto={alto_init})")

im_filtered = ax[2].imshow(ImageFiltered, cmap='gray')
ax[2].set_title(f"Imagen Filtrada (Ancho={ancho_init}, Alto={alto_init})")
plt.colorbar(im_filtered, ax=ax[2])


ax_ancho = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor='lightgray')
ancho_slider = Slider(ax_ancho, 'Ancho', 1, NoiseArray.shape[1], valinit=ancho_init, valstep=1)

ax_alto = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
alto_slider = Slider(ax_alto, 'Alto', 1, NoiseArray.shape[0], valinit=alto_init, valstep=1)

def update(val):
    ancho = int(ancho_slider.val)
    alto = int(alto_slider.val)
    filtro_actualizado = filtro_rectangular(tamaño=NoiseArray.shape, ancho=ancho, alto=alto)
    filtrado_actualizado = filtro_actualizado * ImageFrequencies
    ImageSpectrum_actualizado = np.abs(filtrado_actualizado)
    ImageFiltered_actualizada = np.abs(np.fft.ifft2(np.fft.fftshift(filtrado_actualizado)))

    im_spectrum.set_data(np.log10(1 + ImageSpectrum_actualizado))
    im_filtered.set_data(ImageFiltered_actualizada)
    
    ax[1].set_title(f"Espectro Filtrado (Ancho={ancho}, Alto={alto})")
    ax[2].set_title(f"Imagen Filtrada (Ancho={ancho}, Alto={alto})")
    fig.canvas.draw_idle()

ancho_slider.on_changed(update)
alto_slider.on_changed(update)

plt.show()




def onda_sinusoidal_2d(posiciones_frecuenciales):
    tamaño=(768, 768)
    h, w = tamaño
    x = np.linspace(-w//2, w//2,h)
    y = np.linspace(-h//2, h//2,h)
    X, Y = np.meshgrid(x, y)
    Fun = 0
    for i in range (0,len(posiciones_frecuenciales)):
        Fun += 10*np.sin(2 * np.pi * ( (384 - posiciones_frecuenciales[i][1])* X + (384-posiciones_frecuenciales[i][0])* Y))    # (322,409) (freq 25, -62)
      
    return Fun

posiciones_frecuenciales = [[323,409],[366,425],[372,415]]

A = onda_sinusoidal_2d(posiciones_frecuenciales=posiciones_frecuenciales)

Espectro = np.abs(np.fft.fftshift(np.fft.fft2(A)))

LogEspectro = np.log10(1+Espectro) > 2
Mask = 1-LogEspectro
MaskedSpectrum = ImageFrequencies*Mask

plt.imshow(np.log10((np.abs(MaskedSpectrum))**2+1), cmap='gray')
plt.title('a')
plt.show()

plt.imshow(np.abs(np.fft.ifft2(MaskedSpectrum)), cmap='gray')
plt.title('')
plt.show()