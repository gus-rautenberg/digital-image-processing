from PIL import Image
import argparse

def segment_pixel(value):
    """
    Retorna o novo valor do pixel de acordo com a tabela:
      0-50   -> 25
      51-100 -> 75
      101-150-> 125
      151-200-> 175
      201-255-> 255
    """
    if 0 <= value <= 50:
        return 25
    elif 51 <= value <= 100:
        return 75
    elif 101 <= value <= 150:
        return 125
    elif 151 <= value <= 200:
        return 175
    elif 201 <= value <= 255:
        return 255
    else:
        return value  

def segment_image(image):

    width, height = image.size
    segmented = Image.new("L", (width, height))
    
    pixels_in = image.load()
    pixels_out = segmented.load()
    
    for y in range(height):
        for x in range(width):
            original_value = pixels_in[x, y]
            new_value = segment_pixel(original_value)
            pixels_out[x, y] = new_value
    return segmented

def main():
    parser = argparse.ArgumentParser(
        description="Segmentação de imagem em greyscale sem uso de NumPy. "
                    "Transforma a intensidade dos pixels conforme a tabela especificada."
    )
    parser.add_argument("imagem", help="Caminho para a sua foto.")
    args = parser.parse_args()
    
    img = Image.open(args.imagem)
    gray = img.convert("L")
    
    segmented = segment_image(gray)
    
    gray.show(title="Imagem em Greyscale")
    segmented.show(title="Imagem Segmentada")
    segmented.save("../images/greyscale_segmented.png")

if __name__ == "__main__":
    main()
