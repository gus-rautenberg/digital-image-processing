import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parâmetros fixos
GAUSS_SIZE = 9    # Tamanho do kernel Gaussiano
SIGMA = 1.4      # Valor de sigma para o filtro Gaussiano
THRESHOLD = 0.01  # Limiar para detecção de zero-crossing

def load_image(image_path):
    """
    Carrega uma imagem em escala de cinza e a converte para um array NumPy.
    """
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)

def show_images(original, edges):
    """
    Exibe a imagem original e a imagem processada lado a lado.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Bordas (Zero-Crossing do LoG)")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def convolve(image, kernel):
    """
    Convolui a imagem com o kernel fornecido usando scipy.
    """
    from scipy.ndimage import convolve
    return convolve(image, kernel, mode='constant', cval=0.0)

def gaussian_kernel(size, sigma):
    """
    Gera um kernel Gaussiano n x n.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def laplacian_filter(image):
    """
    Aplica a máscara Laplaciana.
    """
    laplacian_mask = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    return convolve(image, laplacian_mask)

def zero_crossing(image, threshold=0.0):
    """
    Detecta zero-crossings na imagem do LoG.
    """
    rows, cols = image.shape
    zc = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = image[i-1:i+2, j-1:j+2]
            min_val, max_val = patch.min(), patch.max()
            if min_val < -threshold and max_val > threshold:
                zc[i, j] = 255
    return zc

def main():
    parser = argparse.ArgumentParser(description="Detecção de bordas via LoG (Gaussian + Laplacian + Zero Crossing)")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    args = parser.parse_args()
    
    # Carrega a imagem original
    original = load_image(args.imagem)
    
    # 1) Filtragem Gaussiana
    g_kernel = gaussian_kernel(GAUSS_SIZE, SIGMA)
    smoothed = convolve(original, g_kernel)
    
    # 2) Laplaciano da imagem suavizada
    log_image = laplacian_filter(smoothed)
    
    # 3) Detecção de zero-crossings
    edges = zero_crossing(log_image, THRESHOLD)
    
    show_images(original, edges)

if __name__ == '__main__':
    main()
