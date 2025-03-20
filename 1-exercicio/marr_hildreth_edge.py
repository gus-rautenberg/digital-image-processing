import argparse
import math
from PIL import Image
import matplotlib.pyplot as plt
# python marr_hildreth_edge.py ../images/greyscale_segmented.jpg --sigma 1.4 --kernel_size 9 --threshold 0.01
def log_kernel(size, sigma):
    """
    Gera um kernel do Laplaciano do Gaussiano (LoG) sem numpy.
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    kernel = []
    sum_total = 0

    for y in range(-half, half + 1):
        row = []
        for x in range(-half, half + 1):
            norm = (x**2 + y**2 - 2*(sigma**2)) / (sigma**4)
            gaussian = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            value = norm * gaussian
            row.append(value)
            sum_total += value
        kernel.append(row)
    
    mean_value = sum_total / (size * size)
    for y in range(size):
        for x in range(size):
            kernel[y][x] -= mean_value
    
    return kernel

def convolve(image, kernel):
    """
    Convolui manualmente a imagem com o kernel fornecido.
    """
    image_w, image_h = len(image[0]), len(image)
    kernel_w, kernel_h = len(kernel[0]), len(kernel)
    pad_w, pad_h = kernel_w // 2, kernel_h // 2
    
    padded = [[0] * (image_w + 2 * pad_w) for _ in range(image_h + 2 * pad_h)]
    for i in range(image_h):
        for j in range(image_w):
            padded[i + pad_h][j + pad_w] = image[i][j]
    
    output = [[0] * image_w for _ in range(image_h)]
    for i in range(image_h):
        for j in range(image_w):
            value = 0
            for ki in range(kernel_h):
                for kj in range(kernel_w):
                    value += padded[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = value
    return output

def zero_crossing(image, threshold=0.0):
    """
    Detecta zero-crossings na imagem resultante da convolução.
    """
    rows, cols = len(image), len(image[0])
    zc = [[0] * cols for _ in range(rows)]
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            patch = [image[i-1][j-1], image[i-1][j], image[i-1][j+1],
                     image[i][j-1], image[i][j], image[i][j+1],
                     image[i+1][j-1], image[i+1][j], image[i+1][j+1]]
            
            min_val, max_val = min(patch), max(patch)
            if min_val < -threshold and max_val > threshold:
                zc[i][j] = 255
    return zc

def marr_hildreth_edge_detection(image, sigma=1.0, kernel_size=9, threshold=0.0):
    """
    Implementa a detecção de bordas de Marr-Hildreth.
    """
    kernel = log_kernel(kernel_size, sigma)
    log_image = convolve(image, kernel)
    edges = zero_crossing(log_image, threshold)
    return edges

def load_image(image_path):
    """
    Carrega uma imagem em escala de cinza e a converte para uma matriz de inteiros.
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
    plt.title("Bordas - Marr-Hildreth")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Detecção de bordas utilizando o algoritmo de Marr-Hildreth.")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    parser.add_argument("--sigma", type=float, default=1, help="Valor de sigma para o Gaussiano (padrão: 1).")
    parser.add_argument("--kernel_size", type=int, default=9, help="Tamanho do kernel LoG (padrão: 9, deve ser ímpar).")
    parser.add_argument("--threshold", type=float, default=0.01, help="Limiar para detecção de zero-crossing (padrão: 0.01).")
    
    args = parser.parse_args()
    
    img_matrix = load_image(args.imagem)
    edges = marr_hildreth_edge_detection(img_matrix, sigma=args.sigma, kernel_size=args.kernel_size, threshold=args.threshold)
    
    show_images(img_matrix, edges)

if __name__ == '__main__':
    main()
