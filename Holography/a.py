import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import cv2
from scipy.ndimage import label, center_of_mass


""" Definición de parámetros en el plano de entrada y su relación con el plano de salida"""
#---------------------------------------
Nx = 2048  # pixel camara data number
Ny = 2048 
#M = -10                    # Magnificación
#f2 = 200E-3                # Longitud lente de tubo  TL
#f1 = f2/(-M)               # Distancia focal del MO
lamb = 632.80E-9            # Longitud de onda HeNe, rojo
outPixel = 3.45E-6          # Camera pixel 
#inputPixel = (2.74E-6)     # El pixel cómo se ve en la entrada. Afectando el FOV.
#NA = 0.25                  # Apertura numérica del objetivo de microscopio sin(theta)

""" Críterio de resolución de Abbe """

#abbe = lamb/(2*NA)

ruta_imagen = r"C:/Users/Usuario/Desktop/GitHub/entrega3_instrumentos_opticos/Hologram.tiff"
holograma = Image.open(ruta_imagen).convert('L')

hologram = np.array(holograma)  # array de holograma

hologram_ft = np.fft.fftshift( np.fft.fft2(hologram) )  # array de la ft del holograma

hologram_spectrum_log = np.uint8(255*np.log10(1+(np.abs(hologram_ft))**2)/np.max(np.log10(1+(np.abs(hologram_ft))**2)))    # array del espectro en escala log10  del holograma

binary = hologram_spectrum_log > 220

freq_pixel_x = 1/(Nx*outPixel)                      # tamaño de pixel de las frecuencias x
freq_pixel_y = 1/(Ny*outPixel)                      # tamaño de pixel de las freq y
fx = freq_pixel_x * np.arange(-Nx // 2, Nx // 2)    #espacio fx
fy = freq_pixel_x * np.arange(-Ny // 2, Ny // 2)    #espacio fy

#--------------------------------------

# meshgrid of the input and output spaces
x = outPixel * np.arange(-Nx // 2, Nx // 2)
y = outPixel * np.arange(-Ny // 2, Ny // 2)
X, Y = np.meshgrid(x, y)

extent_espacio=[np.min(x),np.max(x),np.min(y),np.max(x)]
extent_frecuencia=[np.min(fx),np.max(fx),np.min(fy),np.max(fy)]


""" Plots: 1 fila y 3 columnas """

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

im1 = axs[0].imshow(hologram, extent=extent_espacio, origin='lower', cmap='gray')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].set_ylabel("")
fig.colorbar(im1, ax=axs[0]) 

im2 = axs[1].imshow(hologram_spectrum_log, extent=extent_frecuencia, origin='lower', cmap='gray')
axs[1].set_title("")
axs[1].set_xlabel("")
fig.colorbar(im2, ax=axs[1])

im3 = axs[2].imshow(binary, extent=extent_frecuencia, origin='lower', cmap='gray')
axs[2].set_title("")
axs[2].set_xlabel("")
fig.colorbar(im3, ax=axs[2]) 

plt.tight_layout()
plt.show()


"""Para seleccionar la frecuencia a la que corresponde el orden +1 o -1"""

def etiquetado(binary,spectrum):
    spectrum=np.uint8(spectrum)
    if len(spectrum.shape) < 3:
        spectrum = cv2.cvtColor(spectrum,cv2.COLOR_GRAY2RGB)
    
    regiones, features = label(binary)

    centers = center_of_mass(binary, regiones, range(1, features + 1))
    print(features)
    print(centers)
    for i in range(0,len(centers)):

        y=np.int32(centers[i][0])
        x=np.int32(centers[i][1])
        
        lado = 25

        x1square=x-lado/2
        x1square = x1square if x1square > 0 else 1
        y1square=y+lado/2
        y1square = y1square if y1square > 0 else 1
        x2square=x+lado/2
        x2square = x2square if x2square > 0 else 1
        y2square= y-lado/2
        y2square = y2square if y2square > 0 else 1

        pos1=np.uint((x1square,y1square))
        pos2=np.uint((x2square,y2square))


        thickness = 3
        color = (255, 0, 0)    
        A=cv2.rectangle(spectrum,pos1,pos2,color,thickness)
        

    plt.imshow(A,cmap='gray', origin='lower')# extent=extent_frecuencia, origin='lower')
    plt.show()
    return A

etiquetado(binary,hologram_spectrum_log)