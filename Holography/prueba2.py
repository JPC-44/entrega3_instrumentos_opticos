import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import cv2
import os 
# Definir parámetros de la imagen
Nx = 2048
Ny = 2048
lamb = 632.80E-9            # Longitud de onda HeNe, rojo
outPixel = 3.45E-6          # Camera pixel 
x = outPixel * np.arange(-Nx // 2, Nx // 2)
y = outPixel * np.arange(-Ny // 2, Ny // 2)
X, Y = np.meshgrid(x, y)


def recortar(imagen):


    cx_init, cy_init = Nx // 2, Ny // 2
    w_init, h_init = 50, 50 

    imagen_recortada_global = np.zeros_like(imagen)
    mascara_global = np.zeros_like(imagen)

    def actualizar(val):
        """Actualiza la imagen con la región recortada."""
        nonlocal imagen_recortada_global, mascara_global
        cx = int(slider_cx.val)
        cy = int(slider_cy.val)
        w = int(slider_w.val)
        h = int(slider_h.val)
        
        mascara = np.zeros_like(imagen)
        
        x1, x2 = max(0, cx - w // 2), min(Nx, cx + w // 2)
        y1, y2 = max(0, cy - h // 2), min(Ny, cy + h // 2)
        
        mascara[y1:y2, x1:x2] = 1
        imagen_recortada_global = imagen * mascara  
        mascara_global = mascara  

        # Actualizar la imagen recortada y el recuadro
        ax.imshow(imagen_recortada_global, cmap="gray")
        rect.set_xy((x1, y1))
        rect.set_width(x2 - x1)
        rect.set_height(y2 - y1)

        fig.canvas.draw()

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    ax.imshow(imagen, cmap="gray")
    rect = plt.Rectangle((cx_init - w_init // 2, cy_init - h_init // 2), w_init, h_init, edgecolor="red", facecolor="none", linewidth=2)
    ax.add_patch(rect)

    ax_cx = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_cy = plt.axes([0.2, 0.15, 0.65, 0.03])
    ax_w = plt.axes([0.2, 0.05, 0.65, 0.03])
    ax_h = plt.axes([0.2, 0.2, 0.65, 0.03])

    slider_cx = Slider(ax_cx, "X Centro", 0, Nx, valinit=cx_init)
    slider_cy = Slider(ax_cy, "Y Centro", 0, Ny, valinit=cy_init)
    slider_w = Slider(ax_w, "Ancho", 10, Nx, valinit=w_init)
    slider_h = Slider(ax_h, "Alto", 10, Ny, valinit=h_init)

    # Conectar sliders con la función de actualización
    slider_cx.on_changed(actualizar)
    slider_cy.on_changed(actualizar)
    slider_w.on_changed(actualizar)
    slider_h.on_changed(actualizar)

    plt.show()

    return imagen_recortada_global, mascara_global

ruta_imagen = r"C:/Users/Usuario/Desktop/GitHub/entrega3_instrumentos_opticos/Hologram.tiff"
holograma = Image.open(ruta_imagen).convert('L')

hologram = np.array(holograma)  

hologram_ft = np.fft.fftshift(np.fft.fft2(hologram))  

hologram_spectrum_log = np.log10(1 + (np.abs(hologram_ft))**2)

guardarespectro =np.uint8(255* hologram_spectrum_log/np.max(hologram_spectrum_log))
cv2.imwrite("espectrolog.png",guardarespectro)

imagen_recortada, mascara = recortar(hologram_spectrum_log)




plt.figure(figsize=(5, 5))
plt.imshow(imagen_recortada, cmap="gray")
plt.title("Imagen Recortada")
plt.axis("off")  # Para quitar los ejes
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(mascara, cmap="gray")
plt.title("Máscara del Recorte")
plt.axis("off") 
plt.show()

""" mask = np.uint8(255*mascara/np.max(mascara))
cv2.imwrite("mascaracentrada.png",mask)
 """


ruta_imagen2 = r"C:/Users/Usuario/Desktop/GitHub/entrega3_instrumentos_opticos/mascara.png"
masc = Image.open(ruta_imagen2).convert('L')

masca = np.array(masc) 

ruta_imagen3 = r"C:/Users/Usuario/Desktop/GitHub/entrega3_instrumentos_opticos/mascaracentrada.png"
maskcentrada = Image.open(ruta_imagen3).convert('L') 

mascaracentrada = np.array(maskcentrada) 

plt.imshow(masca,cmap='gray')
plt.title("espectro")
plt.show()


ft = hologram_ft * masca

m = (255*np.abs(ft)/np.max(np.abs(ft)))
cv2.imwrite("ordenfiltrado.png",m)


imagen1 = np.fft.ifft2(np.fft.fftshift(ft))
onda_plana = np.exp((1j*2*np.pi)*(-Y*62893 + X*98930))

correccion = imagen1 * onda_plana

ftcorreccion = (np.fft.fftshift(np.fft.fft2(correccion))) * mascaracentrada

plt.imshow(np.log10(1+np.abs(ftcorreccion)),cmap='gray')
plt.title("espectro")
plt.show()


plt.imshow(np.abs(correccion),cmap='gray')
plt.title("correc")
plt.show()


holograma = np.fft.ifft2(np.fft.fftshift(ftcorreccion))


def retroprof(hologramaa):

    Campo = hologramaa

    λ = 632.8e-9 
    pixel = 3.45e-6  

    U0 = np.array(Campo, dtype=np.complex128)



    output_folder = "imagenes_propagadas"
    os.makedirs(output_folder, exist_ok=True)


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
        UZ_magnitude = (np.abs(UZ))

        output_path = os.path.join(output_folder, f"propagated_Z_{Z:.5f}m.png")
        plt.imsave(output_path, UZ_magnitude, cmap='gray')
        print(f"Imagen guardada: {output_path}")

    print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")

retroprof(holograma)