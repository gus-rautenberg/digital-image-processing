#!/usr/bin/env python3
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path):
    """Carrega uma imagem em escala de cinza e retorna como array NumPy."""
    img = Image.open(image_path).convert("L")
    return np.array(img)

def otsu_threshold(image):
    """Calcula o limiar de Otsu usando NumPy."""
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    total = histogram.sum()
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
    """Aplica binarização usando NumPy."""
    return np.where(image > threshold, 255, 0).astype(np.uint8)

def erode_image(binary, iterations=1):
    """Erosão usando NumPy com vizinhança 3x3."""
    binary = binary.copy()
    for _ in range(iterations):
        temp = binary.copy()
        for i in range(1, binary.shape[0]-1):
            for j in range(1, binary.shape[1]-1):
                if np.all(temp[i-1:i+2, j-1:j+2] == 255):
                    binary[i, j] = 255
                else:
                    binary[i, j] = 0
    return binary

def dilate_image(binary, iterations=1):
    """Dilatação usando NumPy com vizinhança 3x3."""
    binary = binary.copy()
    for _ in range(iterations):
        temp = binary.copy()
        for i in range(1, binary.shape[0]-1):
            for j in range(1, binary.shape[1]-1):
                if np.any(temp[i-1:i+2, j-1:j+2] == 255):
                    binary[i, j] = 255
                else:
                    binary[i, j] = 0
    return binary

def morphological_opening(binary, erosion_iter=1, dilation_iter=1):
    """Abertura morfológica: erosão seguida de dilatação."""
    eroded = erode_image(binary, iterations=erosion_iter)
    opened = dilate_image(eroded, iterations=dilation_iter)
    return opened

def morphological_closing(binary, erosion_iter=1, dilation_iter=1):
    """Abertura morfológica: erosão seguida de dilatação."""
    opened = dilate_image(binary, iterations=dilation_iter)
    eroded = erode_image(opened, iterations=erosion_iter)
    return eroded

# Função para contar objetos usando flood fill, com tamanho mínimo
def count_objects(binary, min_size=50):
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=int)
    current_label = 1
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Primeira passagem: atribuição de rótulos
    for i in range(height):
        for j in range(width):
            if binary[i, j] == 255:
                neighbors = []
                if j > 0 and binary[i, j-1] == 255:
                    neighbors.append(labels[i, j-1])
                if i > 0 and binary[i-1, j] == 255:
                    neighbors.append(labels[i-1, j])
                if i > 0 and j > 0 and binary[i-1, j-1] == 255:
                    neighbors.append(labels[i-1, j-1])
                if i > 0 and j < width - 1 and binary[i-1, j+1] == 255:
                    neighbors.append(labels[i-1, j+1])
                
                if not neighbors:
                    labels[i, j] = current_label
                    parent[current_label] = current_label
                    current_label += 1
                else:
                    min_label = min(neighbors)
                    labels[i, j] = min_label
                    for lab in neighbors:
                        if lab != min_label:
                            union(min_label, lab)

    label_map = {}
    new_label = 1
    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0:
                root = find(labels[i, j])
                if root not in label_map:
                    label_map[root] = new_label
                    new_label += 1
                labels[i, j] = label_map[root]

    counts = {}
    for label_val in np.unique(labels):
        if label_val == 0:
            continue  # ignora o fundo
        count_pixels = (labels == label_val).sum()
        counts[label_val] = count_pixels

    valid_count = sum(1 for cnt in counts.values() if cnt >= min_size)
    return valid_count

# Função para exibir as imagens em uma grade 2x5
def show_images(original, otsu_binarized, otsu_abertura1, otsu_abertura2, otsu_final, otsu_count,
                manual_binarized, manual_abertura1, manual_abertura2, manual_final, manual_count):
    """Exibe todas as etapas em uma grade de 2 linhas e 5 colunas."""
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    
    # Primeira linha: Otsu
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title("Imagem Original")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(otsu_binarized, cmap='gray')
    axs[0, 1].set_title("Binarizada Otsu")
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(otsu_abertura1, cmap='gray')
    axs[0, 2].set_title("Abertura")
    axs[0, 2].axis('off')
    
    axs[0, 3].imshow(otsu_abertura2, cmap='gray')
    axs[0, 3].set_title("Erosão")
    axs[0, 3].axis('off')
    
    axs[0, 4].imshow(otsu_final, cmap='gray')
    axs[0, 4].set_title(f"Imagem Final (Objetos: {otsu_count})")
    axs[0, 4].axis('off')
    
    # Segunda linha: Manual
    axs[1, 0].imshow(original, cmap='gray')
    axs[1, 0].set_title("Imagem Original")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(manual_binarized, cmap='gray')
    axs[1, 1].set_title("Binarizada Manual (T=128)")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(manual_abertura1, cmap='gray')
    axs[1, 2].set_title("Abertura")
    axs[1, 2].axis('off')
    
    axs[1, 3].imshow(manual_abertura2, cmap='gray')
    axs[1, 3].set_title("Erosão")
    axs[1, 3].axis('off')
    
    axs[1, 4].imshow(manual_final, cmap='gray')
    axs[1, 4].set_title(f"Imagem Final (Objetos: {manual_count})")
    axs[1, 4].axis('off')
    
    plt.tight_layout()
    plt.show()

# Função principal
def main():
    parser = argparse.ArgumentParser(description="Processamento de imagem passo a passo.")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    args = parser.parse_args()
    
    # Carrega a imagem original
    original = load_image(args.imagem)
    
    # Binarização Otsu
    otsu_thresh = otsu_threshold(original)
    otsu_binarized = apply_threshold(original, otsu_thresh)
    otsu_abertura1 = morphological_opening(otsu_binarized, erosion_iter=1, dilation_iter=1)
    # otsu_abertura2 = morphological_opening(otsu_abertura1, erosion_iter=1, dilation_iter=1)
    otsu_abertura2 = erode_image(otsu_abertura1, 1)
    otsu_abertura3 = morphological_closing(otsu_abertura2, 1)

    otsu_final = otsu_abertura3
    otsu_count = count_objects(otsu_final, min_size=50)
    
    # Binarização Manual (T=128)
    manual_thresh = 128
    manual_binarized = apply_threshold(original, manual_thresh)
    manual_abertura1 = morphological_opening(manual_binarized, erosion_iter=1, dilation_iter=1)
    # manual_abertura2 = erode_image(manual_abertura1, 1)
    manual_abertura2 = erode_image(manual_abertura1, 1)
    manual_abertura3 = morphological_closing(manual_abertura2, 1)


    manual_final = manual_abertura3
    manual_count = count_objects(manual_final, min_size=50)
    
    # Exibe todas as etapas
    show_images(original, otsu_binarized, otsu_abertura1, otsu_abertura2, otsu_final, otsu_count,
                manual_binarized, manual_abertura1, manual_abertura2, manual_final, manual_count)

if __name__ == '__main__':
    main()