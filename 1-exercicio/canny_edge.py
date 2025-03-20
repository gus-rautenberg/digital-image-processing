import argparse
import math
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Carrega uma imagem em escala de cinza e a converte para uma matriz de inteiros.
    """
    img = Image.open(image_path).convert("L")
    return [[img.getpixel((x, y)) for x in range(img.width)] for y in range(img.height)]

def save_image(matrix, output_path):
    """
    Salva uma matriz (lista de listas) como imagem.
    """
    height = len(matrix)
    width = len(matrix[0])
    img = Image.new("L", (width, height))
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), int(matrix[y][x]))
    img.save(output_path)

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
    plt.title("Bordas - Canny")
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
    
    padded = [[0] * (image_w + 2 * pad_w) for _ in range(image_h + 2 * pad_h)]
    for i in range(image_h):
        for j in range(image_w):
            padded[i + pad_h][j + pad_w] = image[i][j]
    
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
    Gera um kernel Gaussiano.
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    kernel = []
    sum_total = 0.0
    for y in range(-half, half + 1):
        row = []
        for x in range(-half, half + 1):
            value = (1 / (2 * math.pi * sigma * sigma)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))
            row.append(value)
            sum_total += value
        kernel.append(row)
    # Normaliza o kernel
    for y in range(size):
        for x in range(size):
            kernel[y][x] /= sum_total
    return kernel

def compute_gradient(image):
    """
    Calcula os gradientes (magnitude e ângulo) usando operadores Sobel.
    """
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    sobel_y = [[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]]
    
    Gx = convolve(image, sobel_x)
    Gy = convolve(image, sobel_y)
    
    height = len(image)
    width = len(image[0])
    gradient = [[0]*width for _ in range(height)]
    angle = [[0]*width for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            gx = Gx[i][j]
            gy = Gy[i][j]
            grad = math.sqrt(gx*gx + gy*gy)
            gradient[i][j] = grad
            theta = math.degrees(math.atan2(gy, gx)) if (gx != 0 or gy != 0) else 0
            if theta < 0:
                theta += 180
            angle[i][j] = theta
    return gradient, angle

def non_maximum_suppression(gradient, angle):
    """
    Aplica supressão de não-máximos para afinar as bordas.
    """
    height = len(gradient)
    width = len(gradient[0])
    result = [[0]*width for _ in range(height)]
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            theta = angle[i][j]
            mag = gradient[i][j]
            # Quantiza o ângulo para 4 direções
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                neighbor1 = gradient[i][j-1]
                neighbor2 = gradient[i][j+1]
            elif 22.5 <= theta < 67.5:
                neighbor1 = gradient[i-1][j+1]
                neighbor2 = gradient[i+1][j-1]
            elif 67.5 <= theta < 112.5:
                neighbor1 = gradient[i-1][j]
                neighbor2 = gradient[i+1][j]
            else:  # 112.5 <= theta < 157.5
                neighbor1 = gradient[i-1][j-1]
                neighbor2 = gradient[i+1][j+1]
            if mag >= neighbor1 and mag >= neighbor2:
                result[i][j] = mag
            else:
                result[i][j] = 0
    return result

def double_threshold(image, low, high):
    """
    Aplica limiar duplo à imagem para separar bordas fortes e fracas.
    """
    height = len(image)
    width = len(image[0])
    strong = 255
    weak = 75
    result = [[0]*width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            pixel = image[i][j]
            if pixel >= high:
                result[i][j] = strong
            elif pixel >= low:
                result[i][j] = weak
            else:
                result[i][j] = 0
    return result, strong, weak

def hysteresis(image, strong, weak):
    """
    Aplica histerese para conectar bordas fracas que estejam ligadas a bordas fortes.
    """
    height = len(image)
    width = len(image[0])
    for i in range(1, height-1):
        for j in range(1, width-1):
            if image[i][j] == weak:
                if (image[i-1][j-1] == strong or image[i-1][j] == strong or image[i-1][j+1] == strong or
                    image[i][j-1] == strong or image[i][j+1] == strong or
                    image[i+1][j-1] == strong or image[i+1][j] == strong or image[i+1][j+1] == strong):
                    image[i][j] = strong
                else:
                    image[i][j] = 0
    return image

def canny_edge_detection(image, sigma, kernel_size, low_thresh, high_thresh):
    """
    Implementa a detecção de bordas de Canny.
    """
    # Suavização Gaussiana
    g_kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = convolve(image, g_kernel)
    
    # Cálculo dos gradientes
    gradient, angle = compute_gradient(smoothed)
    
    # Supressão de não-máximos
    nms = non_maximum_suppression(gradient, angle)
    
    # Limiar duplo
    dt, strong, weak = double_threshold(nms, low_thresh, high_thresh)
    
    # Histerese
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
    edges = canny_edge_detection(img_matrix, sigma=args.sigma, kernel_size=args.kernel_size,
                                   low_thresh=args.low, high_thresh=args.high)
    
    show_images(img_matrix, edges)
    
if __name__ == '__main__':
    main()
