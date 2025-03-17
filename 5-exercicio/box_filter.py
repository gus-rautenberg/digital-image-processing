import argparse
from PIL import Image, ImageTk
import tkinter as tk

def box_filter(image, ksize):
    """
    Aplica o filtro box (média) com um kernel de tamanho ksize x ksize.
    A função percorre cada pixel da imagem e calcula a média dos pixels na
    vizinhança definida pelo kernel. Para pixels próximos à borda, usa-se 
    replicação da borda (valores são "esticados").
    """
    width, height = image.size
    # Cria uma nova imagem para o resultado, no modo greyscale ("L")
    output = Image.new("L", (width, height))
    
    # Obtém objetos para acesso aos pixels
    pixels_in = image.load()
    pixels_out = output.load()
    
    # Define o deslocamento para centralizar o kernel
    offset = ksize // 2
    
    for y in range(height):
        for x in range(width):
            total = 0
            count = ksize * ksize
            # Percorre a vizinhança do pixel
            for dy in range(ksize):
                for dx in range(ksize):
                    # Define as coordenadas do pixel na vizinhança
                    ix = x - offset + dx
                    iy = y - offset + dy
                    # Replicação das bordas: se estiver fora, usa o valor da borda
                    if ix < 0:
                        ix = 0
                    elif ix >= width:
                        ix = width - 1
                    if iy < 0:
                        iy = 0
                    elif iy >= height:
                        iy = height - 1
                    total += pixels_in[ix, iy]
            # Calcula a média e atribui o novo valor ao pixel de saída
            avg = total // count
            pixels_out[x, y] = avg
            
    return output

def main():
    parser = argparse.ArgumentParser(
        description="Aplica filtro box (2x2, 3x3, 5x5, 7x7) em uma imagem em greyscale e exibe as imagens."
    )
    parser.add_argument("imagem", help="Caminho para a imagem (em greyscale ou que será convertida).")
    args = parser.parse_args()
    
    # Abre a imagem e converte para greyscale (modo "L")
    image = Image.open(args.imagem).convert("L")
    
    # Aplica os filtros box com os tamanhos solicitados
    filtered_2 = box_filter(image, 2)
    filtered_3 = box_filter(image, 3)
    filtered_5 = box_filter(image, 5)
    filtered_7 = box_filter(image, 7)
    
    # Cria uma janela usando Tkinter para exibir as imagens
    root = tk.Tk()
    root.title("Imagem Original e Filtros Box")
    
    # Cria objetos PhotoImage para cada imagem (usando ImageTk)
    photo_original = ImageTk.PhotoImage(image)
    photo_2 = ImageTk.PhotoImage(filtered_2)
    photo_3 = ImageTk.PhotoImage(filtered_3)
    photo_5 = ImageTk.PhotoImage(filtered_5)
    photo_7 = ImageTk.PhotoImage(filtered_7)
    
    # Organiza as imagens em uma grade
    # Linha 0: Original, Filtro 2x2, Filtro 3x3
    # Linha 1: Filtro 5x5, Filtro 7x7
    # Rótulos
    tk.Label(root, text="Original").grid(row=0, column=0, padx=5, pady=5)
    tk.Label(root, text="Filtro Box 2x2").grid(row=0, column=1, padx=5, pady=5)
    tk.Label(root, text="Filtro Box 3x3").grid(row=0, column=2, padx=5, pady=5)
    tk.Label(root, text="Filtro Box 5x5").grid(row=1, column=0, padx=5, pady=5)
    tk.Label(root, text="Filtro Box 7x7").grid(row=1, column=1, padx=5, pady=5)
    
    # Imagens
    tk.Label(root, image=photo_original).grid(row=0, column=0, padx=5, pady=(25,5))
    tk.Label(root, image=photo_2).grid(row=0, column=1, padx=5, pady=(25,5))
    tk.Label(root, image=photo_3).grid(row=0, column=2, padx=5, pady=(25,5))
    tk.Label(root, image=photo_5).grid(row=1, column=0, padx=5, pady=5)
    tk.Label(root, image=photo_7).grid(row=1, column=1, padx=5, pady=5)
    
    # Se desejar, pode deixar uma célula vazia para manter a grade
    tk.Label(root, text="").grid(row=1, column=2, padx=5, pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()
