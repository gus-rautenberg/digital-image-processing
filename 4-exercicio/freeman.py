import cv2
import argparse
from PIL import Image

# Direções da Cadeia de Freeman (sentido horário)
DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, -1),
              (-1, 0), (-1, 1), (0, 1), (1, 1)]

# DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def load_binary_image(image_path):
    """Carrega e converte a imagem para binária (preto e branco)."""
    image = Image.open(image_path).convert("L")  # Converte para escala de cinza
    width, height = image.size
    binary_image = [[0] * width for _ in range(height)]
    
    # Aplicar limiarização manualmente (threshold = 128)
    for y in range(height):
        for x in range(width):
            binary_image[y][x] = 255 if image.getpixel((x, y)) > 128 else 0
    
    return binary_image, width, height

def find_start_point(image, width, height):
    """Encontra o primeiro pixel da borda do objeto na imagem."""
    for y in range(height):
        for x in range(width):
            if image[y][x] == 255:  # Primeiro pixel branco encontrado
                return x, y
    return None

def freeman_chain(image, width, height):
    """Aplica o algoritmo da Cadeia de Freeman para rastrear a borda."""
    start = find_start_point(image, width, height)
    if not start:
        print("Nenhuma borda encontrada!")
        return []

    x, y = start
    chain_code = []
    prev_dir = 0  # Direção inicial

    while True:
        found = False
        for i in range(8):
            dir_index = (prev_dir + i) % 8  # Percorre as direções no sentido horário
            dx, dy = DIRECTIONS[dir_index]
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and image[ny][nx] == 255:
                chain_code.append(dir_index)
                image[y][x] = 0  # Marca como visitado
                x, y = nx, ny
                prev_dir = (dir_index + 6) % 8  # Ajusta a direção para continuar a borda
                found = True
                break

        if not found or (x, y) == start:
            break  # Sai se não encontrar mais pixels ou se voltar ao início

    return chain_code

def main():
    parser = argparse.ArgumentParser(description="Algoritmo da Cadeia de Freeman em uma imagem binária.")
    parser.add_argument("imagem", help="Caminho para a imagem binária")
    args = parser.parse_args()

    # Carregar e processar a imagem
    image, width, height = load_binary_image(args.imagem)

    # Aplicar o Algoritmo da Cadeia de Freeman
    chain = freeman_chain(image, width, height)

    # Exibir resultado
    print("Código da Cadeia de Freeman:", "".join(map(str, chain)))

if __name__ == "__main__":
    main()
