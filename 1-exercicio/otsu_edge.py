import argparse
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Carrega uma imagem em escala de cinza e a converte para uma matriz (lista de listas) de intensidades.
    """
    img = Image.open(image_path).convert("L")
    width, height = img.size
    return [[img.getpixel((x, y)) for x in range(width)] for y in range(height)]

def compute_histogram(image):
    """
    Calcula o histograma da imagem (valores de 0 a 255).
    """
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

def otsu_threshold(image):
    """
    Calcula o melhor limiar para a imagem utilizando o método de Otsu.
    
    Retorna:
      - threshold: valor de limiar calculado.
    """
    histogram = compute_histogram(image)
    total = sum(histogram)
    
    sum_total = 0
    for i in range(256):
        sum_total += i * histogram[i]
    
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
        
        # Variância entre classes
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t

    return threshold

def apply_threshold(image, threshold):
    """
    Aplica a binarização na imagem utilizando o limiar calculado.
    """
    height = len(image)
    width = len(image[0])
    bin_image = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            bin_image[i][j] = 255 if image[i][j] > threshold else 0
    return bin_image

def show_images(original, binarized, threshold):
    """
    Exibe a imagem original e a imagem binarizada lado a lado.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Imagem Binarizada (Otsu: T = {threshold})")
    plt.imshow(binarized, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Segmentação de imagem usando o método de Otsu.")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    args = parser.parse_args()
    
    # Carrega a imagem
    img_matrix = load_image(args.imagem)
    
    # Calcula o limiar de Otsu
    threshold = otsu_threshold(img_matrix)
    print(f"Limiar calculado: {threshold}")
    
    # Aplica a binarização
    binarized = apply_threshold(img_matrix, threshold)
    
    # Exibe as imagens
    show_images(img_matrix, binarized, threshold)

if __name__ == '__main__':
    main()
