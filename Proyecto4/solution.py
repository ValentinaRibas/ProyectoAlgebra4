import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotarImagen(image, angle):
    x, y, z = image.shape
    M = cv2.getRotationMatrix2D((y / 2, x / 2), angle, 1)
    img_rotada = cv2.warpAffine(image, M, (y, x))
    return img_rotada

def escalarImagen(image, sx, sy):
    x, y, z = image.shape
    img_escalada = cv2.resize(image, (int(y * sx), int(x * sy)))
    return img_escalada

def deformarImagen(image, dx, dy):
    x, y, z = image.shape
    M = np.float32([[1, dx, 0], [dy, 1, 0]])
    img_deformada = cv2.warpAffine(image, M, (y, x))
    return img_deformada


def comprimirImagen(image, k):
    img_comprimida = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[2]):
        U, S, V = np.linalg.svd(image[:, :, i], full_matrices=False)
        U_arr = U[:, :k]
        S_arr = np.diag(S[:k])
        V_arr = V[:k, :]

        img_comprimida[:, :, i] = np.dot(U_arr, np.dot(S_arr, V_arr))

    return img_comprimida

######TRANSFORMACIONES#####
image = cv2.imread('logo.png')

img_escalada1 = rotarImagen(image, 30)  
img_rotada1 = escalarImagen(image, 1.5, 0.5) 
img_deformada1 = deformarImagen(image, 0.2, 0.3) 
img_escalada2 = rotarImagen(image, 80)  
img_rotada2 = escalarImagen(image, 0.5, 1.5) 
img_deformada2 = deformarImagen(image, 0.5, 0.2) 

plt.subplot(1, 3, 1), plt.imshow(image), plt.title('Imagen original:')
plt.subplot(1, 3, 2), plt.imshow(img_escalada1), plt.title('a) Rotada 1:')
plt.subplot(1, 3, 3), plt.imshow(img_escalada2), plt.title('Rotada 2:')
plt.show()
plt.subplot(1, 3, 1), plt.imshow(img_rotada1), plt.title('b) Escalada 1:')
plt.subplot(1, 3, 2), plt.imshow(img_rotada2), plt.title('Escalada 2:')
plt.show()
plt.subplot(1, 3, 1), plt.imshow(img_deformada1), plt.title('c) Deformada 1:')
plt.subplot(1, 3, 2), plt.imshow(img_deformada2), plt.title('Deformada 2:')
plt.show()


######COMPRESION DE IMAGENES#####

img_comprimida1 = comprimirImagen(image, 5)
img_comprimida2 = comprimirImagen(image, 10)
img_comprimida3 = comprimirImagen(image, 30)
img_comprimida4 = comprimirImagen(image, 80)
plt.subplot(1, 4, 1), plt.imshow(img_comprimida1), plt.title('Comprimida 1')
plt.subplot(1, 4, 2), plt.imshow(img_comprimida2), plt.title('Comprimida 2')
plt.subplot(1, 4, 3), plt.imshow(img_comprimida3), plt.title('Comprimida 3')
plt.subplot(1, 4, 4), plt.imshow(img_comprimida4), plt.title('Comprimida 4')
plt.show()

