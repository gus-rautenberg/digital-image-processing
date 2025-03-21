import argparse
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

def box_filter(image, ksize):
    width, height = image.size
    output = Image.new("L", (width, height))
    pixels_in = image.load()
    pixels_out = output.load()
    
    offset = ksize // 2

    for y in range(height):
        for x in range(width):
            total = 0
            count = 0
            for dy in range(-offset, offset + 1):
                for dx in range(-offset, offset + 1):
                    ix = max(0, min(width - 1, x + dx))
                    iy = max(0, min(height - 1, y + dy))
                    total += pixels_in[ix, iy]
                    count += 1
            pixels_out[x, y] = total // count

    return output

def save_images(image, filtered_images, filename):
    image.save("original.png")
    kernel_sizes = [2, 3, 5, 7]

    for i, img in enumerate(filtered_images):
        img.save(f"filtro_{kernel_sizes[i]}x{kernel_sizes[i]}.png")

def display_images(image, filtered_images):
    root = tk.Tk()
    root.title("Filtros Box - Organização Dinâmica")

    canvas = tk.Canvas(root)
    scroll_y = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    scroll_x = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
    
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Atualizar scrollbars
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    images = [image] + filtered_images
    labels = ["Original", "Filtro 2x2", "Filtro 3x3", "Filtro 5x5", "Filtro 7x7"]

    photos = [ImageTk.PhotoImage(img) for img in images]

    max_width = root.winfo_screenwidth() - 100  # Largura máxima disponível
    row, col, row_width = 0, 0, 0

    for i, (label, photo) in enumerate(zip(labels, photos)):
        img_width = photo.width() + 20  
    
        if row_width + img_width > max_width:  
            row += 1
            col = 0
            row_width = 0

        frame_widget = tk.Frame(frame)
        frame_widget.grid(row=row, column=col, padx=10, pady=10)

        tk.Label(frame_widget, text=label, font=("Arial", 12, "bold")).pack()
        tk.Label(frame_widget, image=photo).pack()

        col += 1
        row_width += img_width  
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Aplica filtros Box em uma imagem em greyscale.")
    parser.add_argument("imagem", help="Caminho para a imagem")
    args = parser.parse_args()

    image = Image.open(args.imagem).convert("L")
    filtered_images = [box_filter(image, k) for k in [2, 3, 5, 7]]

    save_images(image, filtered_images, args.imagem)
    display_images(image, filtered_images)

if __name__ == "__main__":
    main()
