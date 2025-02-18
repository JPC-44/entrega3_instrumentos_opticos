import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import cv2
from scipy.ndimage import label, center_of_mass
import os



def retroprof(real,imag):
    # Importar máscara para Z=

    """     image2 = Image.open('imagenreal.png').convert('L')  # Convertir a escala de grises
    image2 = np.array(image2, dtype=np.float64)

    image3 = Image.open('imagenimagin.png').convert('L')  # Convertir a escala de grises
    image3 = np.array(image3, dtype=np.float64) """

    Campo = real + 1j * imag

    λ = 632.8e-9  # Longitud de onda en metros
    pixel = 3.45e-6  # Tamaño de píxel en metros

    #U0 = np.array(image, dtype=np.float64)  # Convertir imagen a float64 para mayor precisión
    U0 = np.array(Campo, dtype=np.complex128)  # Usar tipo complejo



    output_folder = "imagenes_propagadas"
    os.makedirs(output_folder, exist_ok=True)


    UZ_magnitude = (np.abs(U0))

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

        A0 = np.fft.fftshift(np.fft.fft2(U0))
        Az = A0 * H
        Uz = np.fft.ifft2(np.fft.fftshift(Az))

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


    start_Z = get_float_input("Ingrese el valor inicial de Z (en metros): ")
    step_Z = get_float_input("Ingrese el paso entre valores de Z (en metros): ")
    num_images = get_int_input("Ingrese la cantidad de imágenes a generar: ")


    output_folder = "imagenes_propagadas"
    os.makedirs(output_folder, exist_ok=True)


    for i in range(num_images):
        Z = start_Z + i * step_Z
        UZ = AngularSpectrum(U0, Z, λ, pixel)
        UZ_magnitude = (np.abs(UZ))**2

        output_path = os.path.join(output_folder, f"propagated_Z_{Z:.5f}m.png")
        plt.imsave(output_path, UZ_magnitude, cmap='gray')
        print(f"Imagen guardada: {output_path}")

    print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")


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
        
        lado = 70

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

    plane_wave_freq_x = freq_pixel_x*(-centers[2][0]+Nx/2)
    plane_wave_freq_y = freq_pixel_y*((centers[2][1]-Ny/2))
    posiciones = (np.uint16(centers[2][1]),np.uint16(centers[2][0]))
    wave_plane_freq = np.sqrt(plane_wave_freq_y**2+plane_wave_freq_x**2)
    
    print("frecuencia en x",plane_wave_freq_x)
    print("frecuencia en y",plane_wave_freq_y)
    print("frecuencia de onda plana", wave_plane_freq)

    """     plt.imshow(A,cmap='gray', extent=extent_frecuencia)
    plt.show() """
    return A,wave_plane_freq,posiciones,plane_wave_freq_x,plane_wave_freq_y

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
FX,FY = np.meshgrid(fx,fy)
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

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

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
plt.show()


"""Para seleccionar la frecuencia a la que corresponde el orden +1 o -1"""




image,freq,posiciones,freqx,freqy = etiquetado(binary,hologram_spectrum_log)


plt.imshow(image)
plt.show()
imagenn = np.uint8(image)
cv2.imwrite("imagenetiquedata.png",imagenn)
"""Cálculo para el ángulo con el que ingresa al sistema"""

"""Se sabe que la fase de la onda plana es la siguiente : 2pi*x*sin(angle)/lambda"""
"""en la transformada de Fourier es un delta desplazado la freq = sin(angle)/lambda"""




angle = np.arcsin( freq * lamb ) 

print("Ángulo en grados",angle*180/np.pi)






"""Aislamiento del orden +1"""



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




isolated = isolate_and_center(hologram_spectrum_log,  (posiciones[0],posiciones[1]), 300, hologram_ft)

plt.imshow(np.log10(1+(np.abs(isolated))**2), cmap='gray')
plt.title("sadasdfdsdfsfd")
plt.show()

imagen1 = np.fft.ifft2(np.fft.fftshift(isolated))
onda_plana = np.exp((1j*2*np.pi)*(-Y*62893.67 + X*98930.03))


correccion =  imagen1 * onda_plana

ftconfirmacion = np.fft.fftshift(np.fft.fft2(correccion))

plt.imshow(np.log10(1+(np.abs(ftconfirmacion))), cmap='gray')
plt.title("hola espectro")
plt.show()

plt.imshow((np.abs(correccion))**2, cmap='gray')
plt.title("hola usaf")
plt.show()

retroprof(np.real(correccion),np.imag(correccion))



""" Creal = np.real(isolated)
Cimag = np.imag(isolated)
center1 = center_image(Creal)
center2 = center_image(Cimag)
#CTotal = center1+1j*center2
CTotal = np.empty(Creal.shape,dtype = np.complex128)
CTotal.real = center1
CTotal.imag = center2


plt.imshow(np.log10(1+(np.abs(CTotal))**2), cmap='gray')
plt.title("sadasdfdsdfsfd")
plt.show()


Guardar = np.fft.ifft2(np.fft.fftshift(CTotal))
Guardar_real =np.uint8( 255*np.real(Guardar)/np.max(np.real(Guardar)))
Guardad_imaginario =np.uint8( 255*np.imag(Guardar)/np.max(np.imag(Guardar)) ) """




























""" cv2.imwrite("imagenreal.png",Guardar_real)
cv2.imwrite("imagenimagin.png",Guardad_imaginario) """






























""" 
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
plt.show() """





























""" Creal = np.real(isolated_image)
Cimag = np.imag(isolated_image)
CTotal = Creal+1j*Cimag

Guardar = np.fft.ifft2(np.fft.fftshift(CTotal))
Guardar_real =np.uint8( 255*np.real(Guardar)/np.max(np.real(Guardar)))
Guardad_imaginario =np.uint8( 255*np.imag(Guardar)/np.max(np.imag(Guardar)) )

cv2.imwrite("imagenreal.png",Guardar_real)
cv2.imwrite("imagenimagin.png",Guardad_imaginario)

plt.imshow((np.abs(Guardar))**2, cmap='gray')
plt.title("guardar")
plt.show() 






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
plt.show() """

