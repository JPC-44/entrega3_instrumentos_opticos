import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar el holograma
holograma = cv2.imread("C:/Users/Usuario/Desktop/GitHub/entrega3_instrumentos_opticos/Hologram.tiff", cv2.IMREAD_GRAYSCALE)

# Aplicar FFT
fft_holo = np.fft.fftshift(np.fft.fft2(holograma))

# Mostrar espectro
plt.figure(figsize=(6,6))
plt.imshow(np.log1p(np.abs(fft_holo)), cmap='gray')
plt.title("Espectro del holograma")
plt.colorbar()
plt.show()

# Filtrar la banda lateral (ajustar coordenadas según tu holograma)
cx, cy = fft_holo.shape[1]//2, fft_holo.shape[0]//2  # Centro del espectro
r = 50  # Radio del filtro
mask = np.zeros_like(fft_holo, dtype=np.uint8)
cv2.circle(mask, (cx+70, cy+70), r, 1, -1)  # Ajustar desplazamiento

# Aplicar filtro
fft_filtered = fft_holo * mask

# Transformada inversa
reconstruido = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
campo_rec = np.abs(reconstruido)  # Amplitud
fase_rec = np.angle(reconstruido)  # Fase

# Mostrar fase reconstruida
plt.figure(figsize=(6,6))
plt.imshow(np.log10(np.abs(fft_filtered)+1), cmap='jet')
plt.title("Fase reconstruida")
plt.colorbar()
plt.show()
