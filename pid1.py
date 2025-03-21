#Gustavo Portela Rautenberg
#Augusto Dal Pra
#Gabriel Jared
from PIL import Image
import numpy as np

def save_image(array, txt):
    pil_image = Image.fromarray(array)
    pil_image.save(txt + '.png')

def binarize(img_path, threshold=128):
    img = Image.open(img_path).convert('L')  
    binary_image = img.point(lambda p: 255 if p >= threshold else 0)  
    return np.array(binary_image) 

def dilatacao(img_matrix):
    i, j = img_matrix.shape
    dilatacao_result = np.zeros((i, j))

    for x in range(1, i - 1):
        for y in range(1, j - 1):
            aux = [
                img_matrix[x - 1][y],
                img_matrix[x][y - 1],
                img_matrix[x][y],
                img_matrix[x][y + 1],
                img_matrix[x + 1][y]
            ]
            dilatacao_result[x][y] = max(aux)

    return dilatacao_result.astype(np.uint8)

def erosao(img_matrix):
    i, j = img_matrix.shape
    erosao_result = np.zeros((i, j))

    for x in range(1, i - 1):
        for y in range(1, j - 1):
            aux = [
                img_matrix[x - 1][y],
                img_matrix[x][y - 1],
                img_matrix[x][y],
                img_matrix[x][y + 1],
                img_matrix[x + 1][y]
            ]
            erosao_result[x][y] = min(aux)

    return erosao_result.astype(np.uint8)

def abertura(img_matrix):
    auxErosao = erosao(img_matrix)
    abertura = dilatacao(auxErosao)
    return abertura

def fechamento(img_matrix):
    auxDilatacao = dilatacao(img_matrix)
    fechamento = erosao(auxDilatacao)
    return fechamento

def process_image(img_path, threshold=128):
    binary_array = binarize(img_path, threshold)

    save_image(binary_array, 'binary_imageNew')
    save_image(erosao(binary_array), 'erosaoNew')
    save_image(dilatacao(binary_array), 'dilatacaoNew')
    save_image(abertura(binary_array), 'aberturaNew')
    save_image(fechamento(binary_array), 'fechamentoNew')

process_image("./wall.png", threshold=128)