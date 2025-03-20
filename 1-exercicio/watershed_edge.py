#!/usr/bin/env python3
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

# Parâmetro: porcentagem do background (em relação ao gradiente máximo)
BACKGROUND_PERCENTAGE = 0.1

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
    plt.title("Bordas (Watershed)")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def image_to_matrix(image):
    """
    Converte a imagem em um array numpy de intensidades.
    """
    return np.array(image)

def compute_gradient(matrix):
    """
    Calcula o gradiente da imagem utilizando a diferença entre os pixels vizinhos.
    """
    grad_x = np.diff(matrix, axis=1, append=0)  # Gradiente no eixo X
    grad_y = np.diff(matrix, axis=0, append=0)  # Gradiente no eixo Y
    grad = np.abs(grad_x) + np.abs(grad_y)      # Gradiente total
    return grad

def get_neighbors(x, y, width, height):
    """
    Retorna os vizinhos (8-conectividade) de um pixel (x, y).
    """
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    return neighbors

def flood_fill(x, y, label, grad, threshold, labels):
    """
    Preenche a região com o rótulo fornecido usando flood fill.
    """
    height, width = grad.shape
    queue = [(x, y)]
    labels[y, x] = label
    while queue:
        cx, cy = queue.pop(0)  # Remoção FIFO
        for nx, ny in get_neighbors(cx, cy, width, height):
            if labels[ny, nx] == -1 and grad[ny, nx] < threshold:
                labels[ny, nx] = label
                queue.append((nx, ny))

def watershed_segmentation(image_path):
    """
    Realiza a segmentação Watershed na imagem.
    """
    # Carregar a imagem
    img = Image.open(image_path)
    if img.mode != "L":
        img = img.convert("L")
    
    matrix = image_to_matrix(img)
    height, width = matrix.shape

    # Calcular o gradiente
    grad = compute_gradient(matrix)
    max_grad = np.max(grad)
    threshold = BACKGROUND_PERCENTAGE * max_grad

    # Inicializa a matriz de rótulos (-1 significa não processado)
    labels = -np.ones((height, width), dtype=int)
    next_label = 1  # labels positivos para regiões, 0 para watershed (bordas)

    # Marcar os marcadores (regiões de background) onde o gradiente é baixo
    for y in range(height):
        for x in range(width):
            if grad[y, x] < threshold and labels[y, x] == -1:
                flood_fill(x, y, next_label, grad, threshold, labels)
                next_label += 1

    # Criar uma lista de pixels não marcados, ordenados pelo valor do gradiente
    pixel_list = [(grad[y, x], x, y) for y in range(height) for x in range(width) if labels[y, x] == -1]
    pixel_list.sort(key=lambda t: t[0])

    # Inundação: define a linha de watershed (bordas) onde há conflito de rótulos
    WATERSHED = 0
    for g, x, y in pixel_list:
        neighbor_labels = set(labels[ny, nx] for nx, ny in get_neighbors(x, y, width, height) if labels[ny, nx] != -1)
        if len(neighbor_labels) == 0:
            continue
        elif len(neighbor_labels) == 1:
            labels[y, x] = neighbor_labels.pop()
        else:
            labels[y, x] = WATERSHED

    # Criar a imagem de saída: pixels com label 0 serão pretos (bordas)
    out_img = Image.fromarray(np.uint8(labels == WATERSHED) * 255)

    # Exibir a imagem original e a imagem de bordas lado a lado
    show_images(img, out_img)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: {} caminho_da_imagem".format(sys.argv[0]))
        sys.exit(1)
    image_path = sys.argv[1]
    watershed_segmentation(image_path)
