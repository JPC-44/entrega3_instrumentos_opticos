import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters
#---------------------------------------
N = 720  # data number
inputPixel = 125 / N  # size / pixels
outPixel = 3.45E-6  # camera pixel
f1 = 100E-3
f2 = 1
M = -f2 / f1
lamb = 632E-9
d = 10
initial_pupil_radius = 1000E-2  # init values
initial_circ2_radius = 1500E-2  
radius1 =1
radius2 = 1
#--------------------------------------

# meshgrid of the input and output spaces
x = inputPixel * np.arange(-N // 2, N // 2)
y = inputPixel * np.arange(-N // 2, N // 2)
X, Y = np.meshgrid(x, y)

u =inputPixel *np.arange(-N // 2, N // 2)
v = inputPixel* np.arange(-N // 2, N // 2)
U, V = np.meshgrid(u, v)



def circ1(x, y, radius):
    a = (x**2 + y**2 < radius**2)
    return a

def circ2(x, y, radius):
    b = (x**2 + y**2 < (2*radius)**2)
    return b

C1= circ1(U,V,1000E-6)
C2 = circ1(U,V,3.5E-3)

plt.imshow(C1+C2,cmap='gray')
plt.show()

# pupila que es la resta de dos circ para generar un anillo binario
def pupil(radius1, radius2):
    b = circ2(X, Y, radius2)   
    a = circ1(X, Y, radius1)  
    c = np.logical_xor(b, a)  
    return 1*c


# lectura de los datos para convertirlos en array
def entrance_image():
    archivo_csv = r"C:/Users/Usuario/Downloads/MuestraBio_E02(2).csv"
    datos = np.loadtxt(archivo_csv, delimiter=',', dtype=str)
    matriz_compleja = np.vectorize(lambda x: complex(x.replace(' ', '').replace('i', 'j')))(datos)
    return matriz_compleja


# update func para los sliders
def update(val):
    global radius1
    global radius2
    radius1 = slider1.val  # lee los valores que se le asignó al slider en pantalla 
    radius2 = slider2.val  # y se asignan a estas dos variables
    pupila = pupil(radius1, radius2) # se actualiza la pupila con estos nuevos radios
    print(radius1,radius2)
    # se hace shift a la pupila para que sea la función de transferencia
    pupila_shifted = np.fft.fftshift(pupila)

    # se calcula el campo de salida 
    output_field = np.fft.ifft2(np.multiply(pupila_shifted, entrance_ft))

    # se actualizan los plots
    im1.set_data(np.abs(pupila))
    im2.set_data(np.log10(1+(np.abs(np.fft.fftshift(np.fft.fft2(output_field)))))) 
    im3.set_data(np.abs(output_field)**2)
    im4.set_data(np.angle(output_field))
    plt.draw()


# campo de entrada
entrance_field = entrance_image()
import cv2
#ft del campo de entrada
entrance_ft = np.fft.fft2(entrance_field)
entrance_ftt = 255*np.log10(1+np.abs(np.fft.fftshift(entrance_ft))**2)/np.max(np.log10(np.abs(np.fft.fftshift(entrance_ft))**2))
cv2.imwrite('amplitud.png',255*np.abs(entrance_field)/np.max(np.abs(entrance_field)))
fase_gaussiana = np.exp(1j * (X**2 + Y**2) / (2 * f1))  

# se evalua la pupila inicial
pupila_fase = pupil(initial_pupil_radius, initial_circ2_radius)
pupila_shifted = np.fft.fftshift(pupila_fase)
output_field = np.fft.ifft2(np.multiply(pupila_fase, entrance_ft))
fase_pupila=np.angle(pupila_fase)




"""---------------------------plots-----------------------"""
fig, axes = plt.subplots(2, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.3)  


im1 = axes[0, 0].imshow(np.abs(pupila_fase), cmap='gray', extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
axes[0, 0].set_title('Pupila')


im2 = axes[0, 1].imshow(np.log10( 1 + (np.abs(np.fft.fftshift(entrance_ft))**2)), cmap='gray')
axes[0, 1].set_title('FT entrada')

im3 = axes[1, 0].imshow(np.abs(output_field), cmap='gray')
axes[1, 0].set_title('Magnitud en salida')

im4 = axes[1, 1].imshow(np.angle(output_field), cmap='gray')
axes[1, 1].set_title('Fase salida')

# Add first slider for circ1 radius
ax_slider1 = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider1 = Slider(ax_slider1, 'Pupil Radius (circ1)', 0, 100, valinit=initial_pupil_radius,valstep=1E-6)
slider1.on_changed(update)

# Add second slider for circ2 radius
ax_slider2 = plt.axes([0.2, 0.12, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider2 = Slider(ax_slider2, 'Radius for circ2',0, 5, valinit=initial_circ2_radius,valstep=1E-6)
slider2.on_changed(update)

plt.show()
"""--------------------------ending plots-----------------------"""

# radius2 = 1.7277
# radius1 = 3
fx = (1E+6)/125 *(np.arange(-720//2,720//2)) 
Fx,Fy = np.meshgrid(fx,fx)

A= np.multiply(np.fft.fftshift(pupil(radius1,radius2)),entrance_ft)
plt.imshow(np.log10(1 + (np.abs(np.fft.fftshift(A))**2)),cmap='gray',extent=[np.min(fx),np.max(fx),np.min(fx),np.max(fx)])
plt.show()
print(A.shape)

lensdiameter = 7E-3
fourierradius = (3.5E-3)/(f1*lamb)
print(fourierradius)