import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# Obtener la ruta de la carpeta donde está el script
current_dir = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_stripe_pattern_v3(image_size=2848, margin=50, vertical_margin=200, group_spacing=50, text_size=3, text_thickness=8):
    """
    Genera un patrón de franjas con márgenes verticales más grandes para mejorar la visibilidad de los números.

    Parámetros:
    - image_size: Tamaño de la imagen (asumida cuadrada).
    - margin: Espacio en píxeles en los bordes laterales.
    - vertical_margin: Espacio en píxeles en los bordes superior e inferior.
    - group_spacing: Espacio adicional entre grupos de 3 pares de franjas.
    - text_size: Tamaño del texto de los números.
    - text_thickness: Grosor de los números.

    Retorna:
    - image: Imagen con el patrón de franjas y los números de grupo.
    """
    # Crear fondo blanco
    image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Coordenadas de inicio después de los márgenes
    x_pos = margin
    y_margin = vertical_margin  # Márgenes más grandes en la parte superior e inferior
    bottom_margin = vertical_margin

    group_counter = 0  # Contador para los grupos de 3
    group_number = 1  # Número de grupo
    group_start_x = x_pos  # Posición inicial del grupo para centrar el número

    # Tamaño inicial de franja y separación
    stripe_width = 1
    separation = 1

    while x_pos + stripe_width * 2 + separation <= image_size - margin:
        # Dibujar primera franja (dentro de los márgenes superior e inferior)
        image[y_margin:image_size - bottom_margin, x_pos:x_pos + stripe_width] = 0
        # Dibujar segunda franja
        image[y_margin:image_size - bottom_margin, x_pos + stripe_width + separation:x_pos + 2 * stripe_width + separation] = 0

        # Mover a la siguiente secuencia con mismo tamaño de franja pero mayor separación
        x_pos += 2 * stripe_width + separation + 5  # Espacio adicional entre pares de franjas
        separation += 1  # Aumentar separación

        # Contar grupos de 3 y agregar el espaciamiento extra
        group_counter += 1
        if group_counter == 3:
            # Calcular posición centrada del número del grupo
            group_center_x = (group_start_x + x_pos) // 2
            text_position = (group_center_x, image_size - (bottom_margin // 2))  # Ubicación dentro del margen inferior
            cv2.putText(image, str(group_number), text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,), text_thickness, cv2.LINE_AA)

            # Actualizar para el siguiente grupo
            x_pos += group_spacing  # Añadir espaciamiento extra entre grupos
            group_start_x = x_pos  # Reiniciar la posición inicial del grupo
            group_counter = 0  # Reiniciar el contador
            group_number += 1  # Incrementar el número del grupo

        # Luego, cambiar el tamaño de las franjas y resetear separación
        if separation > stripe_width + 2:
            stripe_width += 1
            separation = stripe_width

    return image

# Generar la imagen del patrón de franjas corregido
stripe_pattern_v3 = generate_stripe_pattern_v3()

# Mostrar la imagen generada
plt.figure(figsize=(10, 10))
plt.imshow(stripe_pattern_v3, cmap='gray', aspect='auto')
plt.axis('off')
plt.show()



output_path = os.path.join(current_dir, "DIEGO_2025_resolution_test_chart.png")
cv2.imwrite(output_path, stripe_pattern_v3)