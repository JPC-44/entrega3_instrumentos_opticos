# Reimportar librerías después del reset
import numpy as np
import matplotlib.pyplot as plt

# Datos en formato "distancia, ángulo"
datos_texto = """
206, 1
202, 3
201, 5
291, 7
197, 9
197, 10
190, 12
183, 14
185, 16
184, 18
189, 19
192, 21
199, 23
211, 25
212, 27
210, 28
214, 30
205, 32
201, 34
196, 36
185, 37
187, 39
182, 41
176, 43
169, 45
168, 46
172, 48
180, 50
191, 52
202, 54
217, 55
236, 57
262, 59
305, 61
356, 63
400, 64
428, 66
438, 68
434, 70
429, 72
423, 73
417, 75
412, 77
406, 79
401, 81
397, 82
394, 84
396, 86
393, 88
394, 90
391, 91
389, 93
389, 95
389, 97
387, 99
386, 100
384, 102
380, 104
376, 106
373, 108
370, 109
366, 111
363, 113
359, 115
359, 117
358, 118
357, 120
355, 122
354, 124
352, 126
353, 127
354, 129
355, 131
354, 133
356, 135
356, 136
356, 138
358, 140
358, 142
359, 144
360, 145
358, 147
357, 149
356, 151
350, 153
346, 154
342, 156
335, 158
328, 160
326, 162
322, 163
319, 165
315, 167
314, 169
312, 171
309, 172
308, 174
307, 176
304, 178
303, 180
303, 181
302, 183
301, 185
299, 187
300, 189
300, 190
303, 192
305, 194
305, 196
307, 198
309, 199
312, 201
315, 203
318, 205
318, 207
324, 208
328, 210
332, 212
335, 214
342, 216
342, 217
346, 219
345, 221
342, 223
338, 225
332, 226
326, 228
316, 230
307, 232
299, 234
290, 235
284, 237
278, 239
276, 241
274, 243
272, 244
271, 246
271, 248
268, 250
269, 252
267, 253
265, 255
265, 257
263, 259
261, 261
258, 262
253, 264
250, 266
246, 268
243, 270
241, 271
237, 273
235, 275
234, 277
232, 279
231, 280
231, 282
232, 284
"""

# Función para procesar y graficar los datos
def procesar_y_graficar_desde_texto(datos_texto):
    """
    Convierte un bloque de texto con formato "distancia, ángulo" en coordenadas cartesianas
    y las grafica en 2D.

    Parámetros:
    - datos_texto: Texto con múltiples líneas en formato "distancia, ángulo".

    Retorna:
    - Gráfica de la trayectoria en 2D.
    """

    # Dividir el texto en líneas y convertir a tuplas de enteros
    datos = [tuple(map(int, linea.split(','))) for linea in datos_texto.strip().split('\n')]

    # Separar en listas: distancias y ángulos
    distancias = np.array([d[0] for d in datos])
    angulos = np.array([d[1] for d in datos])

    # Convertir ángulos de grados a radianes
    angulos_radianes = np.radians(angulos)

    # Convertir coordenadas polares a cartesianas
    x = distancias * np.cos(angulos_radianes)
    y = distancias * np.sin(angulos_radianes)

    # 📌 Graficar en 2D con dimensiones
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'bo-', label="Trayectoria")  # 'bo-' → círculos azules con línea
    plt.xlabel("Eje X (mm)")
    plt.ylabel("Eje Y (mm)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Línea horizontal
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)  # Línea vertical
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("Coordenadas Polares convertidas a Cartesianas")
    plt.axis('equal')  # Mantener escala 1:1
    plt.show()

# 📌 Ejecutar la función con los datos dados
procesar_y_graficar_desde_texto(datos_texto)
