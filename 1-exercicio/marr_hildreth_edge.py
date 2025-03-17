import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
# python seu_script.py caminho/para/sua_imagem.jpg --sigma 1.4 --kernel_size 9 --threshold 0.01

def log_kernel(size, sigma):
    """
    Gera um kernel do Laplaciano do Gaussiano (LoG).
    
    Parâmetros:
      - size: tamanho do kernel (deve ser ímpar)
      - sigma: desvio padrão do Gaussiano
      
    Retorna:
      - kernel LoG normalizado
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    norm = (x**2 + y**2 - 2*(sigma**2)) / (sigma**4)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = norm * gaussian
    kernel -= kernel.mean()
    return kernel

def convolve(image, kernel):
    """
    Convolui manualmente a imagem com o kernel fornecido.
    
    Parâmetros:
      - image: imagem de entrada (matriz 2D)
      - kernel: kernel para convolução
      
    Retorna:
      - imagem resultante da convolução
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=float)
    
    for i in range(image_h):
        for j in range(image_w):
            regiao = padded[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(regiao * kernel)
    return output

def zero_crossing(image, threshold=0.0):
    """
    Detecta zero-crossings na imagem resultante da convolução.
    
    Parâmetros:
      - image: imagem após aplicação do filtro LoG
      - threshold: limiar para considerar a variação de sinal
      
    Retorna:
      - imagem binária com as bordas detectadas (valor 255 para borda)
    """
    rows, cols = image.shape
    zc = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            patch = image[i-1:i+2, j-1:j+2]
            if (np.any(patch > threshold) and np.any(patch < -threshold)):
                zc[i, j] = 255
    return zc

def marr_hildreth_edge_detection(image, sigma=1.0, kernel_size=9, threshold=0.0):
    """
    Implementa a detecção de bordas de Marr-Hildreth.
    
    Parâmetros:
      - image: imagem de entrada (escala de cinza)
      - sigma: parâmetro do Gaussiano
      - kernel_size: tamanho do kernel LoG
      - threshold: limiar para detecção de zero-crossing
      
    Retorna:
      - imagem binária com as bordas detectadas
    """
    kernel = log_kernel(kernel_size, sigma)
    log_image = convolve(image, kernel)
    edges = zero_crossing(log_image, threshold)
    return edges

def main():
    parser = argparse.ArgumentParser(description="Detecção de bordas utilizando o algoritmo de Marr-Hildreth.")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    parser.add_argument("--sigma", type=float, default=1.4, help="Valor de sigma para o Gaussiano (padrão: 1.4).")
    parser.add_argument("--kernel_size", type=int, default=9, help="Tamanho do kernel LoG (padrão: 9, deve ser ímpar).")
    parser.add_argument("--threshold", type=float, default=0.01, help="Limiar para detecção de zero-crossing (padrão: 0.01).")
    
    args = parser.parse_args()
    
    # Carrega a imagem em escala de cinza
    img = cv2.imread(args.imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar a imagem. Verifique o caminho fornecido.")
        return
    
    edges = marr_hildreth_edge_detection(img, sigma=args.sigma, kernel_size=args.kernel_size, threshold=args.threshold)
    
    # Exibe a imagem original e a imagem com as bordas detectadas
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Bordas - Marr-Hildreth")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
