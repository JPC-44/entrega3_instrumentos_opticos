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
""" cv2.imwrite("hologram.png",hologram_spectrum_log) """
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

x1 = np.arange(-Nx // 2, Nx // 2)
y1 = np.arange(-Ny // 2, Ny // 2)
X1, Y1 = np.meshgrid(x1, y1)

extent_espacio=[np.min(x),np.max(x),np.min(y),np.max(x)]
extent_frecuencia=[np.min(fx),np.max(fx),np.min(fy),np.max(fy)]


""" Plots: 1 fila y 3 columnas """

""" fig, axs = plt.subplots(1, 3, figsize=(12, 4))

im1 = axs[0].imshow(hologram, extent=extent_espacio, cmap='gray')
axs[0].set_title("")
axs[0].set_xlabel("")
axs[0].set_ylabel("")
fig.colorbar(im1, ax=axs[0]) 

im2 = axs[1].imshow(hologram_spectrum_log, extent=extent_frecuencia, cmap='gray')
axs[1].set_title("")
axs[1].set_xlabel("")
fig.colorbar(im2, ax=axs[1])

im3 = axs[2].imshow(binary, extent=extent_frecuencia, cmap='gray')
axs[2].set_title("")
axs[2].set_xlabel("")
fig.colorbar(im3, ax=axs[2]) 

plt.tight_layout()
plt.show() """


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
        

    plt.imshow(A,cmap='gray')# extent=extent_frecuencia)
    plt.show()
    plane_wave_freq_x = freq_pixel_x*(centers[2][0]-Nx/2)
    plane_wave_freq_y = freq_pixel_y*((centers[2][1]-Ny/2))
    posiciones = (np.uint16(centers[2][1]),np.uint16(centers[2][0]))
    wave_plane_freq = np.sqrt(plane_wave_freq_y**2+plane_wave_freq_x**2)
    print("frecuencia en x",plane_wave_freq_x)
    print("frecuencia en y",plane_wave_freq_y)
    print("frecuencia de onda plana", wave_plane_freq)
    return A,wave_plane_freq,posiciones


image,freq,posiciones = etiquetado(binary,hologram_spectrum_log)

"""Cálculo para el ángulo con el que ingresa al sistema"""

"""Se sabe que la fase de la onda plana es la siguiente : 2pi*x*sin(angle)/lambda"""
"""en la transformada de Fourier es un delta desplazado la freq = sin(angle)/lambda"""

angle = np.arcsin( freq * lamb ) 

print("Ángulo en grados",angle*180/np.pi)






"""Aislamiento del orden +1"""

def isolate_and_center(image, center, radius, image_to_isolate):

    """
    Aísla un área circular en la imagen

    Parámetros:
    - image: imagen.
    - center: (x, y) coordenadas del centro del círculo.
    - radius: Radio del círculo.

    Retorna:
    - Imagen procesada con el círculo aislado y centrado.
    """
    # Cargar la imagen en escala de grises
   
    # mascara que con circulo en la posición center
    mask = np.zeros_like(image, dtype=np.uint8)
    a=cv2.circle(mask, center, radius, 255, -1)
    a = a/np.max(a)
    """     plt.imshow(np.multiply(a,image),cmap='gray')
    plt.show() """

    # mascara * imagen para aislar el orden
    isolated_image = np.multiply(a,image_to_isolate)

    return isolated_image

def center_image(isolated_image):
    """
    se ingresa una imagen que tiene un
    ROI el cual se quiere centrar
    """
    """     fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(isolated_image, cmap='gray')
    axs[0].set_title("")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    fig.colorbar(im1, ax=axs[0]) 

    im2 = axs[1].imshow(hologram_spectrum_log, cmap='gray')
    axs[1].set_title("")
    axs[1].set_xlabel("")
    fig.colorbar(im2, ax=axs[1])


    plt.tight_layout()
    plt.show() """


    h, w = isolated_image.shape
    center_x, center_y = w // 2, h // 2

    _, binary = cv2.threshold(isolated_image, 50, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)  

    # contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contorno del circulo
    largest_contour = max(contours, key=cv2.contourArea)

    # bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # posición del circulo
    roi = isolated_image[y:y+h, x:x+w]

    # posición de centro
    new_x = center_x - w // 2
    new_y = center_y - h // 2

    new_img = np.zeros_like(isolated_image)
    new_img[new_y:new_y+h, new_x:new_x+w] = roi

    # con y sin desplazamiento
    """     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(isolated_image, cmap='gray')
    axes[0].set_title("Imagen Original")
    axes[1].imshow(new_img, cmap='gray')
    axes[1].set_title("Imagen con Círculo Centrados")
    plt.show() """

    # guardar
    cv2.imwrite("image_centered.png", new_img)
    return new_img




isolated_image = isolate_and_center(hologram_spectrum_log,  (posiciones[0],posiciones[1]), 350, hologram_ft)







image_with_plane_wave = np.fft.ifft2(np.fft.fftshift(isolated_image))

image_without_plane_wave  = image_with_plane_wave * np.exp(-1j*2*np.pi*X*(np.sin(angle + 80*np.pi/180))/lamb)


plt.imshow(np.angle(np.exp(-1j*2*np.pi*X1*(np.sin(angle + 85*np.pi/180))/lamb)), cmap='gray')
plt.show()


Creal = np.real(isolated_image)
Cimag = np.imag(isolated_image)
C1 = center_image(Creal)
C2 = center_image(Cimag)
CTotal = C1+1j*C2

Guardar = np.fft.ifft2(np.fft.fftshift(CTotal))
Guardar_real =np.uint8( 255*np.real(Guardar)/np.max(np.real(Guardar)))
Guardad_imaginario =np.uint8( 255*np.imag(Guardar)/np.max(np.imag(Guardar)) )

cv2.imwrite("imagenreal.png",Guardar_real)
cv2.imwrite("imagenimagin.png",Guardad_imaginario)




b = np.log10(np.abs(CTotal)+1)
Binary  = 255*b/np.max(b) > 230

Binary = 1-Binary
plt.imshow(Binary, cmap='gray')
plt.show()
A = np.multiply(CTotal,Binary)
fft_aislado1 = np.fft.ifft2(np.fft.fftshift(A))

FTconfirm = np.fft.fftshift(np.fft.fft2(fft_aislado1)) 

plt.imshow(np.log10(1+(np.abs(FTconfirm))**2), cmap='gray')
plt.show()

plt.imshow((np.abs(fft_aislado1))**2, cmap='gray')
plt.title("planewave_spectrum no plane")
plt.show()

