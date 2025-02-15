import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters
#---------------------------------------

Nx = 2048  # data number
Ny = 2448
inputPixelx = 125 / Nx  # size / pixels
inputPixely = 125 / Ny  # size / pixels
outPixel = 3.45E-6  # camera pixel
f1 = 10E-3
f2 = 20E-3
M = -f2 / f1
lamb = 632E-9
d = 10
initial_pupil_radius = 1000E-2 
initial_circ2_radius = 1500E-2
#--------------------------------------


x = inputPixelx * np.arange(-Nx // 2, Nx // 2)
y = inputPixely * np.arange(-Ny // 2, Ny // 2)
X, Y = np.meshgrid(x, y)
u = M * outPixel * np.arange(-Nx // 2, Nx // 2)
v = M * outPixel * np.arange(-Ny // 2, Ny // 2)
U, V = np.meshgrid(u, v)


def circ1(x, y, radius):
    a = (x**2 + y**2 < radius**2)
    return a

def circ2(x, y, radius):
    b = (x**2 + y**2 < (2*radius)**2)
    return b

def pupil(radius1, radius2):
    b = circ2(X, Y, radius2)  
    a = circ1(X, Y, radius1)  
    c = np.logical_xor(b, a) 
    return c

def entrance_image():
    archivo_csv = r"C:/Users/Usuario/Downloads/MuestraBio_E02(2).csv"
    datos = np.loadtxt(archivo_csv, delimiter=',', dtype=str)
    matriz_compleja = np.vectorize(lambda x: complex(x.replace(' ', '').replace('i', 'j')))(datos)

    pad_height = (Ny - matriz_compleja.shape[0]) // 2
    pad_width = (Nx - matriz_compleja.shape[1]) // 2

    pad_height_extra = (Ny - matriz_compleja.shape[0]) % 2
    pad_width_extra = (Nx - matriz_compleja.shape[1]) % 2

    array_padded = np.pad(matriz_compleja, 
    ((pad_height, pad_height + pad_height_extra), 
    (pad_width, pad_width + pad_width_extra)),
    mode='constant', constant_values = 0)

    return array_padded


def update(val):
    radius1 = slider1.val 
    radius2 = slider2.val  
    pupila = pupil(radius1, radius2)  
    pupila_shifted =np.fft.fftshift(pupila)

    output_field = (np.fft.ifft2(np.multiply(pupila_shifted, entrance_ft)))
    # Update plots
    im1.set_data(pupila)
    im2.set_data(np.angle(entrance_field))
    im3.set_data(np.abs(output_field)**2)
    im4.set_data(np.angle(output_field))

    plt.draw()


# 
entrance_field = entrance_image()
entrance_ft = np.fft.fft2(entrance_field)
pupila = pupil(initial_pupil_radius, initial_circ2_radius)  
pupila_shifted = np.fft.fftshift(pupila)
output_field = np.fft.ifft2(np.multiply(pupila_shifted, entrance_ft))
output_ft = np.fft.fftshift(np.fft.fft2(output_field))


fig, axes = plt.subplots(2, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.3)




im1 = axes[0, 0].imshow(pupila, cmap='gray',extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
axes[0, 0].set_title('Pupila')

im2 = axes[0, 1].imshow(np.abs(output_ft), cmap='gray')
axes[0, 1].set_title('ft output')

im3 = axes[1, 0].imshow(np.abs(output_field), cmap='gray')
axes[1, 0].set_title('Magnitud en salida')

im4 = axes[1, 1].imshow(np.angle(output_field), cmap='gray')
axes[1, 1].set_title('Fase salida')


ax_slider1 = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider1 = Slider(ax_slider1, 'Pupil Radius (circ1)', 0, 10, valinit=initial_pupil_radius,valstep=1E-6)
slider1.on_changed(update)


ax_slider2 = plt.axes([0.2, 0.12, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider2 = Slider(ax_slider2, 'Radius for circ2', 0, 10, valinit=initial_circ2_radius,valstep=1E-6)
slider2.on_changed(update)

plt.show()

