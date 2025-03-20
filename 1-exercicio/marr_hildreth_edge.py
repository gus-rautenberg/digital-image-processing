import argparse
import math
from PIL import Image
import matplotlib.pyplot as plt

# Parâmetros fixos
GAUSS_SIZE = 9    # Tamanho do kernel Gaussiano
SIGMA = 1.4      # Valor de sigma para o filtro Gaussiano
THRESHOLD = 0.01  # Limiar para detecção de zero-crossing

def load_image(image_path):
    """
    Carrega uma imagem em escala de cinza e a converte para uma matriz (lista de listas) de intensidades.
    """
    img = Image.open(image_path).convert("L")
    return [[img.getpixel((x, y)) for x in range(img.width)] for y in range(img.height)]

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
    Convolui manualmente a imagem com o kernel fornecido.
    """
    image_h = len(image)
    image_w = len(image[0])
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # Cria uma imagem "padded" com zeros
    padded = [[0] * (image_w + 2 * pad_w) for _ in range(image_h + 2 * pad_h)]
    for i in range(image_h):
        for j in range(image_w):
            padded[i + pad_h][j + pad_w] = image[i][j]
    
    # Convolução
    output = [[0] * image_w for _ in range(image_h)]
    for i in range(image_h):
        for j in range(image_w):
            acc = 0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    acc += padded[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = acc
    return output

def gaussian_kernel(size, sigma):
    """
    Gera um kernel Gaussiano n x n a partir de G(x,y)=exp(-(x²+y²)/(2σ²)).
    Normaliza o kernel para que a soma dos elementos seja 1.
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    kernel = []
    sum_total = 0
    for y in range(-half, half + 1):
        row = []
        for x in range(-half, half + 1):
            value = math.exp(-((x**2 + y**2) / (2 * sigma**2)))
            row.append(value)
            sum_total += value
        kernel.append(row)
    # Normalização
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_total
    return kernel

def laplacian_filter(image):
    """
    Aplica a máscara Laplaciana [1 1 1; 1 -8 1; 1 1 1] na imagem.
    """
    laplacian_mask = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]
    return convolve(image, laplacian_mask)

def zero_crossing(image, threshold=0.0):
    """
    Detecta zero-crossings na imagem do LoG.
    Um pixel é considerado borda se, em sua vizinhança 3x3, houver valores menores que -threshold e maiores que threshold.
    """
    rows = len(image)
    cols = len(image[0])
    zc = [[0] * cols for _ in range(rows)]
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = [
                image[i-1][j-1], image[i-1][j], image[i-1][j+1],
                image[i][j-1],   image[i][j],   image[i][j+1],
                image[i+1][j-1], image[i+1][j], image[i+1][j+1]
            ]
            min_val = min(patch)
            max_val = max(patch)
            if min_val < -threshold and max_val > threshold:
                zc[i][j] = 255
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
