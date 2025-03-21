import argparse
from PIL import Image

def convert_to_grayscale(image_path, output_path):
    """
    Converte uma imagem colorida para escala de cinza manualmente.
    
    Parâmetros:
      - image_path: Caminho para a imagem de entrada
      - output_path: Caminho para salvar a imagem em escala de cinza
    """
    # Carrega a imagem
    img = Image.open(image_path)
    img = img.convert('RGB')  # Garante que seja uma imagem RGB
    
    # Obtém os pixels da imagem
    pixels = img.load()
    width, height = img.size
    
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            pixels[i, j] = (gray, gray, gray)
    
    img.save(output_path)
    print(f"Imagem convertida para escala de cinza e salva em: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Converte uma imagem para escala de cinza sem OpenCV.")
    parser.add_argument("input", help="Caminho para a imagem de entrada.")
    parser.add_argument("output", help="Caminho para salvar a imagem convertida.")
    
    args = parser.parse_args()
    convert_to_grayscale(args.input, args.output)
    
if __name__ == "__main__":
    main()
