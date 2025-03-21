import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import sys

# --- Funções Compartilhadas ---
def load_image(image_path):
    """Carrega uma imagem em escala de cinza e a converte para um array NumPy."""
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)

def gaussian_kernel(size, sigma):
    """Gera um kernel Gaussiano n x n."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    """Convolui manualmente a imagem com o kernel fornecido usando NumPy."""
    kernel = np.array(kernel)
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    
    return output

# --- Algoritmo Marr-Hildreth ---
def laplacian_filter(image):
    """Aplica a máscara Laplaciana."""
    laplacian_mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return convolve(image, laplacian_mask)

def zero_crossing(image, threshold=0.0):
    """Detecta zero-crossings na imagem do LoG."""
    rows, cols = image.shape
    zc = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = image[i-1:i+2, j-1:j+2]
            min_val, max_val = patch.min(), patch.max()
            if min_val < -threshold and max_val > threshold:
                zc[i, j] = 255
    return zc

def marr_hildreth(image, gauss_size=9, sigma=1.4, threshold=0.04):
    """Implementa o algoritmo Marr-Hildreth (LoG)."""
    g_kernel = gaussian_kernel(gauss_size, sigma)
    smoothed = convolve(image, g_kernel)
    log_image = laplacian_filter(smoothed)
    edges = zero_crossing(log_image, threshold)
    return edges

# --- Algoritmo Canny ---
def sobel_filters():
    """Retorna os kernels de Sobel para gradiente em x e y."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sobel_x, sobel_y

def compute_gradient(image):
    """Calcula os gradientes (magnitude e ângulo) usando operadores Sobel."""
    sobel_x, sobel_y = sobel_filters()
    Gx = convolve(image, sobel_x)
    Gy = convolve(image, sobel_y)
    magnitude = np.hypot(Gx, Gy)
    angle = np.arctan2(Gy, Gx) * (180 / np.pi) % 180
    return magnitude, angle

def non_maximum_suppression(gradient, angle):
    """Afina as bordas por supressão de não-máximos."""
    height, width = gradient.shape
    result = np.zeros_like(gradient)
    
    angle = (angle / 45).astype(int) % 4
    for i in range(1, height-1):
        for j in range(1, width-1):
            q, r = 255, 255
            if angle[i, j] == 0:
                q, r = gradient[i, j-1], gradient[i, j+1]
            elif angle[i, j] == 1:
                q, r = gradient[i-1, j+1], gradient[i+1, j-1]
            elif angle[i, j] == 2:
                q, r = gradient[i-1, j], gradient[i+1, j]
            elif angle[i, j] == 3:
                q, r = gradient[i-1, j-1], gradient[i+1, j+1]
            
            if gradient[i, j] >= q and gradient[i, j] >= r:
                result[i, j] = gradient[i, j]
    
    return result

def double_threshold(image, low, high):
    """Aplica limiar duplo para separar bordas fortes e fracas."""
    strong, weak = 255, 75
    result = np.zeros_like(image)
    result[image >= high] = strong
    result[(image >= low) & (image < high)] = weak
    return result, strong, weak

def hysteresis(image, strong, weak):
    """Conecta bordas fracas a fortes usando histerese."""
    height, width = image.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if image[i, j] == weak:
                if np.any(image[i-1:i+2, j-1:j+2] == strong):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image, sigma=1.0, kernel_size=5, low_thresh=20, high_thresh=40):
    """Implementa a detecção de bordas de Canny."""
    smoothed = convolve(image, gaussian_kernel(kernel_size, sigma))
    gradient, angle = compute_gradient(smoothed)
    nms = non_maximum_suppression(gradient, angle)
    dt, strong, weak = double_threshold(nms, low_thresh, high_thresh)
    final_edges = hysteresis(dt, strong, weak)
    return final_edges

# --- Algoritmo Otsu ---
def compute_histogram(image):
    """Calcula o histograma da imagem (valores de 0 a 255)."""
    return np.histogram(image, bins=np.arange(257))[0]

def otsu_threshold(image):
    """Calcula o melhor limiar para a imagem utilizando o método de Otsu."""
    histogram = compute_histogram(image)
    total = image.size
    sum_total = np.sum(np.arange(256) * histogram)
    
    sum_b = 0
    weight_b = 0
    var_max = 0
    threshold = 0
    
    for t in range(256):
        weight_b += histogram[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * histogram[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t

    return threshold

def apply_threshold(image, threshold):
    """Aplica a binarização na imagem utilizando o limiar calculado."""
    return np.where(image > threshold, 255, 0)

# --- Algoritmo Watershed ---
def watershed_segmentation(image, background_percentage=0.05, foreground_threshold=0.3, sigma=2, kernel_size=5):
    """Segmenta uma imagem em escala de cinza usando o algoritmo Watershed."""
    # 1. Pré-processamento
    gaussian = gaussian_kernel(kernel_size, sigma)
    smoothed = convolve(image, gaussian)
    
    # 2. Calcular o gradiente usando Sobel
    sobel_x, sobel_y = sobel_filters()
    grad_x = convolve(smoothed, sobel_x)
    grad_y = convolve(smoothed, sobel_y)
    gradient = np.hypot(grad_x, grad_y)
    
    # 3. Definir marcadores
    foreground_markers = np.zeros_like(image, dtype=int)
    foreground_markers[image > foreground_threshold * image.max()] = 2  
    
    height, width = image.shape
    border_size = int(min(height, width) * background_percentage)
    background_markers = np.zeros_like(image, dtype=int)
    background_markers[:border_size, :] = 1  # Topo
    background_markers[-border_size:, :] = 1  # Fundo
    background_markers[:, :border_size] = 1  # Esquerda
    background_markers[:, -border_size:] = 1  # Direita
    
    markers = np.zeros_like(image, dtype=int)
    markers[background_markers == 1] = 1  
    markers[foreground_markers == 2] = 2  
    
    # 4. Implementar o Watershed com flood fill
    labels = markers.copy()
    queue = deque()
    
    for i in range(height):
        for j in range(width):
            if labels[i, j] == 1:
                queue.append((i, j))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width:
                if labels[nx, ny] == 0:  
                    labels[nx, ny] = labels[x, y]
                    queue.append((nx, ny))
    
    labels[labels == 0] = 1  
    return labels

# --- Função Principal para Plotagem ---
def main():
    parser = argparse.ArgumentParser(description="Detecção de bordas e segmentação com múltiplos algoritmos.")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    args = parser.parse_args()
    
    original = load_image(args.imagem)
    
    marr_hildreth_edges = marr_hildreth(original, gauss_size=9, sigma=1.4, threshold=0.01)
    canny_edges = canny_edge_detection(original, sigma=1.0, kernel_size=5, low_thresh=20, high_thresh=40)
    threshold = otsu_threshold(original)
    otsu_binarized = apply_threshold(original, threshold)
    watershed_labels = watershed_segmentation(original, background_percentage=0.05, foreground_threshold=0.3, sigma=2, kernel_size=5)
    
    plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 2, (1, 2))
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 3)
    plt.title("Marr-Hildreth")
    plt.imshow(marr_hildreth_edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.title("Canny")
    plt.imshow(canny_edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 5)
    plt.title(f"Otsu (T = {threshold})")
    plt.imshow(otsu_binarized, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.title("Watershed")
    plt.imshow(watershed_labels, cmap='jet')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

#O Marr-Hildreth utiliza a derivada de Laplace para realçar regiões de transição de intensidade, 
# identificando bordas através da detecção de zero-crossings, o que pode resultar em contornos mais espessos
#  e sensíveis a ruídos. Em contrapartida, o Canny emprega os filtros de Sobel para calcular os gradientes 
# e suas magnitudes, seguido de uma supressão de não-máximos para afinar os contornos, e aplica um duplo limiar 
# (histerese) para classificar as bordas em fortes e fracas, produzindo resultados com bordas mais precisas e bem definidas.
#  Apesar de sua implementação ser mais complexa, o Canny tende a oferecer melhor performance em termos de delimitação dos contornos
#  quando comparado ao Marr-Hildreth, que possui uma abordagem mais simples, porém menos robusta frente a variações e ruídos na imagem.