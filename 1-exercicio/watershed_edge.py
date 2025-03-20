import argparse
from PIL import Image
import math
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)
#python watershed.py caminho/para/sua_imagem.jpg --method distance --marker_param 0.7
#python watershed.py caminho/para/sua_imagem.jpg --method percent --marker_param 0.8

#####################################
# Funções básicas de manipulação de imagem
#####################################
def load_image(image_path):
    """
    Carrega uma imagem em escala de cinza e retorna uma matriz (lista de listas) de intensidades.
    """
    img = Image.open(image_path).convert("L")
    width, height = img.size
    return [[img.getpixel((x, y)) for x in range(width)] for y in range(height)]

def show_images(original, segmentation, title="Segmentação Watershed"):
    """
    Exibe a imagem original e a imagem segmentada lado a lado.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(segmentation, cmap='nipy_spectral')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#####################################
# Otsu para binarização (utilizado na abordagem de distância)
#####################################
def compute_histogram(image):
    histogram = [0]*256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

def otsu_threshold(image):
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
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold

def binarize(image, threshold):
    """
    Converte a imagem em uma imagem binária: 1 para foreground, 0 para background.
    """
    height = len(image)
    width = len(image[0])
    binary = [[0]*width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            binary[i][j] = 1 if image[i][j] > threshold else 0
    return binary

#####################################
# Transformação de Distância (usando distância Manhattan)
#####################################
def distance_transform(binary):
    """
    Calcula a distância Manhattan de cada pixel do foreground ao background.
    Utiliza uma abordagem de dois passes.
    """
    height = len(binary)
    width = len(binary[0])
    max_val = width + height
    dist = [[max_val if binary[i][j]==1 else 0 for j in range(width)] for i in range(height)]
    
    # Passagem para frente
    for i in range(height):
        for j in range(width):
            if dist[i][j] == 0:
                continue
            if i > 0:
                dist[i][j] = min(dist[i][j], dist[i-1][j] + 1)
            if j > 0:
                dist[i][j] = min(dist[i][j], dist[i][j-1] + 1)
    # Passagem para trás
    for i in range(height-1, -1, -1):
        for j in range(width-1, -1, -1):
            if i < height-1:
                dist[i][j] = min(dist[i][j], dist[i+1][j] + 1)
            if j < width-1:
                dist[i][j] = min(dist[i][j], dist[i][j+1] + 1)
    return dist

#####################################
# Labeling de componentes conexos (usado para agrupar marcadores)
#####################################
def label_markers(binary_marker):
    """
    Rótula componentes conexos (4-conexos) na imagem binária de marcadores utilizando DFS iterativa.
    Retorna uma matriz com rótulos (números inteiros) e o número de rótulos encontrados.
    """
    height = len(binary_marker)
    width = len(binary_marker[0])
    labels = [[0]*width for _ in range(height)]
    current_label = 1

    for i in range(height):
        for j in range(width):
            if binary_marker[i][j] == 1 and labels[i][j] == 0:
                # DFS iterativa utilizando uma pilha
                stack = [(i, j)]
                labels[i][j] = current_label
                while stack:
                    ci, cj = stack.pop()
                    # Vizinhança 4-conexa
                    for ni, nj in [(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)]:
                        if 0 <= ni < height and 0 <= nj < width:
                            if binary_marker[ni][nj] == 1 and labels[ni][nj] == 0:
                                labels[ni][nj] = current_label
                                stack.append((ni, nj))
                current_label += 1
    return labels, current_label - 1

#####################################
# Estratégias para definição de marcadores
#####################################
def markers_by_distance(image, marker_percent):
    """
    Define marcadores com base na distância: 
      - Binariza a imagem usando Otsu.
      - Calcula a transformação de distância.
      - Define como marcador os pixels cuja distância seja >= marker_percent * max_distance.
      - Agrupa os marcadores em componentes conexos.
    """
    thresh = otsu_threshold(image)
    binary = binarize(image, thresh)
    dist = distance_transform(binary)
    height = len(dist)
    width = len(dist[0])
    max_d = 0
    for i in range(height):
        for j in range(width):
            if dist[i][j] > max_d:
                max_d = dist[i][j]
    # Define marcadores: 1 se distância >= marker_percent * max, 0 caso contrário.
    marker_binary = [[1 if dist[i][j] >= marker_percent * max_d else 0 for j in range(width)] for i in range(height)]
    markers, num = label_markers(marker_binary)
    print(f"Marcadores (distance-based): {num} encontrados (marker_percent = {marker_percent})")
    return markers

def markers_by_percent(image, percent_threshold):
    """
    Define marcadores com base na intensidade: pixels com valor > (percent_threshold*255) são marcados.
    Em seguida, agrupa os marcadores em componentes conexos.
    """
    height = len(image)
    width = len(image[0])
    marker_binary = [[1 if image[i][j] >= percent_threshold * 255 else 0 for j in range(width)] for i in range(height)]
    markers, num = label_markers(marker_binary)
    print(f"Marcadores (percent-based): {num} encontrados (percent_threshold = {percent_threshold*100}%)")
    return markers

#####################################
# Region Growing Simplificado (Watershed)
#####################################
def region_growing_watershed(markers):
    """
    Propaga os rótulos dos marcadores para toda a imagem utilizando uma abordagem iterativa simples.
    Pixels não marcados (0) recebem o rótulo se todos os seus vizinhos (8-conexos) que já possuem rótulo
    forem iguais; caso haja conflito, o pixel recebe -1 (região de watershed).
    Itera até convergência.
    """
    height = len(markers)
    width = len(markers[0])
    labels = [row[:] for row in markers]  # copia inicial
    changed = True
    while changed:
        changed = False
        for i in range(height):
            for j in range(width):
                if labels[i][j] == 0:  # não rotulado
                    neighbor_labels = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni = i + di
                            nj = j + dj
                            if ni < 0 or ni >= height or nj < 0 or nj >= width:
                                continue
                            if labels[ni][nj] != 0:
                                neighbor_labels.append(labels[ni][nj])
                    if neighbor_labels:
                        # Se todos os vizinhos têm o mesmo rótulo, atribui esse rótulo; senão, marca como borda (-1)
                        unique = set(neighbor_labels)
                        if len(unique) == 1:
                            labels[i][j] = neighbor_labels[0]
                            changed = True
                        else:
                            labels[i][j] = -1
                            changed = True
    return labels

#####################################
# Função principal do Watershed
#####################################
def watershed_segmentation(image, method, marker_param):
    """
    Executa a segmentação Watershed utilizando uma das estratégias de marcadores:
      - method: 'distance' ou 'percent'
      - marker_param: marker_percent (ex: 0.7) se method=='distance'
                      ou percent_threshold (ex: 0.8) se method=='percent'
    """
    if method == 'distance':
        markers = markers_by_distance(image, marker_param)
    elif method == 'percent':
        markers = markers_by_percent(image, marker_param)
    else:
        raise ValueError("Método inválido. Use 'distance' ou 'percent'.")
    
    segmentation = region_growing_watershed(markers)
    return segmentation

#####################################
# Main
#####################################
def main():
    parser = argparse.ArgumentParser(description="Segmentação Watershed (método simplificado).")
    parser.add_argument("imagem", help="Caminho para a imagem em escala de cinza.")
    parser.add_argument("--method", choices=['distance', 'percent'], default='distance',
                        help="Método para definição dos marcadores: 'distance' ou 'percent' (padrão: distance).")
    parser.add_argument("--marker_param", type=float, default=0.7,
                        help=("Parâmetro para os marcadores: se method=='distance', "
                              "é a fração do valor máximo da distância (padrão: 0.7); "
                              "se method=='percent', é o limiar percentual (0-1) da intensidade (padrão: 0.7)."))
    args = parser.parse_args()
    
    # Carrega a imagem
    img = load_image(args.imagem)
    
    # Executa o watershed
    segmentation = watershed_segmentation(img, args.method, args.marker_param)
    
    # Exibe o resultado
    show_images(img, segmentation, title=f"Watershed ({args.method}-based)")

if __name__ == '__main__':
    main()
