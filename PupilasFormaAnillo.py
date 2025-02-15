import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
N = 720
inputPixel = 125 / N
outPixel = 3.45E-6
f1 = 10
f2 = 10
M = -f2 / f1
lamb = 632E-9
d = 10
radius1 = 1000E-2
radius2 = 1500E-2
Nx = 2048
Ny = 2448

x = inputPixel * np.arange(-N // 2, N // 2)
y = inputPixel * np.arange(-N // 2, N // 2)
X, Y = np.meshgrid(x, y)
u =-outPixel * np.arange(-Nx // 2, Nx// 2)
v = -outPixel * np.arange(-Ny // 2, Ny // 2)


def circ1(x, y, radius):
    return x**2 + y**2 < radius**2

def circ2(x, y, radius):
    return x**2 + y**2 < (2 * radius)**2

def pupil(radius1, radius2):
    b = circ2(X, Y, radius2)
    a = circ1(X, Y, radius1)
    return np.logical_xor(b, a)

def entrance_image():
    archivo_csv = r"C:/Users/Usuario/Downloads/MuestraBio_E02(2).csv"
    datos = np.loadtxt(archivo_csv, delimiter=',', dtype=str)
    return np.vectorize(lambda x: complex(x.replace(' ', '').replace('i', 'j')))(datos)

def update(val):
    global radius1, radius2
    radius1 = slider1.val
    radius2 = slider2.val
    pupila = pupil(radius1, radius2)
    pupila_shifted = np.fft.fftshift(pupila)
    output_field = np.fft.ifft2(np.multiply(pupila_shifted, entrance_ft))

    def reflect_matrix(U):
        return np.flip(np.flip(U, axis=0), axis=1)
    def center_image(small_img, target_size=(2048, 2448)):
        small_h, small_w = small_img.shape
        target_h, target_w = target_size
        pad_top = (target_h - small_h) // 2
        pad_bottom = target_h - small_h - pad_top
        pad_left = (target_w - small_w) // 2
        pad_right = target_w - small_w - pad_left
        centered_img = np.pad(small_img, ((pad_top, pad_bottom), (pad_left, pad_right)), 
        mode='constant', constant_values=0)
        return centered_img
    
    image_fliped_paded = center_image(reflect_matrix(output_field))
    Fourier_Filtrado = 255*np.log10(1+np.abs((np.fft.fftshift(np.multiply(pupila_shifted, entrance_ft)))))/np.max(np.log10(1+np.abs((np.fft.fftshift(np.multiply(pupila_shifted, entrance_ft))))))
    cv2.imwrite('FourierFIltrado.png',Fourier_Filtrado)

    im_magnitude.set_data((np.abs(image_fliped_paded))**2)
    plt.draw()

entrance_field = entrance_image()
entrance_ft = np.fft.fft2(entrance_field)

pupila_fase = pupil(radius1, radius2)
pupila_shifted = np.fft.fftshift(pupila_fase)
output_field = np.fft.ifft2(np.multiply(pupila_fase, entrance_ft))

fig, ax = plt.subplots(figsize=(7, 6))
plt.subplots_adjust(bottom=0.3)

im_magnitude = ax.imshow(np.abs(output_field), cmap='gray', extent=[np.min(u), np.max(u), np.min(v), np.max(v)])
ax.set_title('Magnitud en salida')
ax.set_xlabel('Eje u (m)')  # Título del eje x
ax.set_ylabel('Eje v (m)')  # Título del eje y

ax_slider1 = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider1 = Slider(ax_slider1, 'Pupil Radius (circ1)', 0, 100, valinit=radius1, valstep=1E-6)
slider1.on_changed(update)

ax_slider2 = plt.axes([0.2, 0.12, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider2 = Slider(ax_slider2, 'Radius for circ2', 0, 5, valinit=radius2, valstep=1E-6)
slider2.on_changed(update)

plt.show()

"""--------------------------ending plots-----------------------"""

# radius2 = 1.7277
# radius1 = 3


""" print(radius1,radius2)



pupila = pupil(radius1, radius2)

pupila_shifted = np.fft.fftshift(pupila)

output_field = np.fft.ifft2(np.multiply(pupila_shifted, entrance_ft))


plt.imshow(np.abs(output_field)**2,cmap='gray')
plt.show() """