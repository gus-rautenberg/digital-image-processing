#!/usr/bin/env python3
from PIL import Image
import sys
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

# Como a imagem já está em grayscale, basta extrair a matriz de intensidades.
def image_to_matrix(image):
    width, height = image.size
    pixels = list(image.getdata())
    matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pixels[y * width + x])
        matrix.append(row)
    return matrix

# Cálculo simples do gradiente: diferença absoluta com o vizinho direito e inferior
def compute_gradient(matrix):
    height = len(matrix)
    width = len(matrix[0])
    grad = [[0]*width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            gx = abs(matrix[y][x+1] - matrix[y][x]) if x < width - 1 else 0
            gy = abs(matrix[y+1][x] - matrix[y][x]) if y < height - 1 else 0
            grad[y][x] = gx + gy
    return grad

# Função para obter vizinhos (8-conectividade)
def get_neighbors(x, y, width, height):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    return neighbors

# Implementação do flood fill "na mão" sem usar deque
def flood_fill(x, y, label, grad, threshold, labels):
    height = len(grad)
    width = len(grad[0])
    queue = [(x, y)]
    labels[y][x] = label
    while queue:
        cx, cy = queue.pop(0)  # Remoção FIFO usando pop(0)
        for nx, ny in get_neighbors(cx, cy, width, height):
            if labels[ny][nx] == -1 and grad[ny][nx] < threshold:
                labels[ny][nx] = label
                queue.append((nx, ny))

def watershed_segmentation(image_path):
    # Carregar a imagem. Como ela já está em grayscale, não há necessidade de converter.
    img = Image.open(image_path)
    if img.mode != "L":
        img = img.convert("L")
    
    matrix = image_to_matrix(img)
    height = len(matrix)
    width = len(matrix[0])

    # Calcular o gradiente da imagem
    grad = compute_gradient(matrix)
    max_grad = max(max(row) for row in grad)
    threshold = BACKGROUND_PERCENTAGE * max_grad

    # Inicializa a matriz de rótulos; -1 significa "não processado"
    labels = [[-1]*width for _ in range(height)]
    next_label = 1  # labels positivos para regiões, 0 para watershed (bordas)

    # Marcar os marcadores (regiões de background) onde o gradiente é baixo
    for y in range(height):
        for x in range(width):
            if grad[y][x] < threshold and labels[y][x] == -1:
                flood_fill(x, y, next_label, grad, threshold, labels)
                next_label += 1

    # Criar uma lista de pixels não marcados, ordenados pelo valor do gradiente
    pixel_list = []
    for y in range(height):
        for x in range(width):
            if labels[y][x] == -1:
                pixel_list.append((grad[y][x], x, y))
    pixel_list.sort(key=lambda t: t[0])

    # Inundação: define a linha de watershed (bordas) onde há conflito de rótulos
    WATERSHED = 0
    for g, x, y in pixel_list:
        neighbor_labels = set()
        for nx, ny in get_neighbors(x, y, width, height):
            if labels[ny][nx] != -1:
                neighbor_labels.add(labels[ny][nx])
        if len(neighbor_labels) == 0:
            continue
        elif len(neighbor_labels) == 1:
            labels[y][x] = neighbor_labels.pop()
        else:
            labels[y][x] = WATERSHED

    # Criar a imagem de saída: pixels com label 0 serão pretos (bordas)
    out_img = Image.new("L", (width, height))
    out_pixels = out_img.load()
    for y in range(height):
        for x in range(width):
            out_pixels[x, y] = 0 if labels[y][x] == WATERSHED else 255

    # Exibir a imagem original e a imagem de bordas lado a lado
    show_images(img, out_img)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: {} caminho_da_imagem".format(sys.argv[0]))
        sys.exit(1)
    image_path = sys.argv[1]
    watershed_segmentation(image_path)
