import numpy as np
import argparse
import math
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """Carrega uma imagem em escala de cinza e a converte para um array NumPy."""
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float32)

def save_image(matrix, output_path):
    """Salva um array NumPy como imagem."""
    img = Image.fromarray(matrix.astype(np.uint8))
    img.save(output_path)

def show_images(original, edges):
    """Exibe a imagem original e a imagem processada lado a lado."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Bordas - Canny")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

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

def gaussian_kernel(size, sigma):
    """Gera um kernel Gaussiano usando NumPy."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def compute_gradient(image):
    """Calcula os gradientes (magnitude e ângulo) usando operadores Sobel."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
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

def canny_edge_detection(image, sigma, kernel_size, low_thresh, high_thresh):
    """Implementa a detecção de bordas de Canny."""
    smoothed = convolve(image, gaussian_kernel(kernel_size, sigma))
    gradient, angle = compute_gradient(smoothed)
    nms = non_maximum_suppression(gradient, angle)
    dt, strong, weak = double_threshold(nms, low_thresh, high_thresh)
    final_edges = hysteresis(dt, strong, weak)
    return final_edges

def main():
    parser = argparse.ArgumentParser(description="Detecção de bordas utilizando o algoritmo de Canny.")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Valor de sigma para o Gaussiano (padrão: 1.0).")
    parser.add_argument("--kernel_size", type=int, default=5, help="Tamanho do kernel Gaussiano (padrão: 5, deve ser ímpar).")
    parser.add_argument("--low", type=float, default=20, help="Limiar inferior para o threshold (padrão: 20).")
    parser.add_argument("--high", type=float, default=40, help="Limiar superior para o threshold (padrão: 40).")
    
    args = parser.parse_args()
    img_matrix = load_image(args.imagem)
    edges = canny_edge_detection(img_matrix, sigma=args.sigma, kernel_size=args.kernel_size, low_thresh=args.low, high_thresh=args.high)
    show_images(img_matrix, edges)
    
if __name__ == '__main__':
    main()
