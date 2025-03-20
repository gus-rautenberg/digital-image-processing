import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

#======================================================================
#                     IMAGEM
#======================================================================

# Abrir a imagem e converter para escala de cinza usando skimage
img = Image.open('../images/3.png')
img_gray = color.rgb2gray(np.array(img))

# Normalizar para valores entre 0 e 255
img_gray = (img_gray * 255).astype(np.uint8)

#======================================================================
#                     IMAGEM
#======================================================================

#======================================================================
#                     Marr-Hildreth
#====================================================================== 

# convolução 2D
def convolve2d(image, kernel):
    # convulação 2d 
    kernel = np.flipud(np.fliplr(kernel))  # Inverter o kernel
    output = np.zeros_like(image, dtype=np.float64)

    # Adicionar padding
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Aplicar convolução, percorre cada pixel e aplica o kernel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]])

    return output

# Aplicar suavização Gaussiana manualmente
def gaussian_blur(image, kernel_size=5, sigma=1.4):
    # filtro gaussiano
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) # calculo gaus
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma)) # calculo gaus
    kernel = np.outer(gauss, gauss) # matriz 2d gaus
    kernel /= np.sum(kernel) # normalização gaus
    
    return convolve2d(image, kernel)

blurred = gaussian_blur(img_gray)

# Aplicar o operador Laplaciano
laplacian_kernel = np.array([[0,  1,  0], 
                             [1, -4,  1], 
                             [0,  1,  0]])

laplacian = convolve2d(blurred, laplacian_kernel)

# Implementação do detector de zero-crossing
def zero_crossing(laplacian):
    # Detecta bordas identificando mudanças de sinal na matriz
    rows, cols = laplacian.shape # matriz para armazenar as bordas
    edges = np.zeros((rows, cols), dtype=np.uint8) # matriz para armazenar as bordas
    
    for i in range(1, rows - 1): # percorre a visiznhança 3x3 anotando as transições de sinais
        for j in range(1, cols - 1):
            neg = pos = False
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if laplacian[i + x, j + y] > 0:
                        pos = True
                    if laplacian[i + x, j + y] < 0:
                        neg = True
            if pos and neg:
                edges[i, j] = 255  # Marca como borda
    return edges

edges = zero_crossing(laplacian)

#======================================================================
#                     Marr-Hildreth
#====================================================================== 

#======================================================================
#                     Canny
#====================================================================== 

#detecção de bordas
def sobel_operator(image):
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                         [0,  0,  0],
                         [1,  2,  1]])
    
    grad_x = convolve2d(image, sobel_x) # bordas verticais 
    grad_y = convolve2d(image, sobel_y) # bordas horizontais

    magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Magnitude do gradiente, força da borda
    direction = np.arctan2(grad_y, grad_x)  # Direção do gradiente, orientação da borda
    
    return magnitude, direction


#  supressão de não maximos
def non_maximum_suppression(magnitude, direction): 
    rows, cols = magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.float64)
    
    angle = direction * 180 / np.pi  # Converter para graus
    angle[angle < 0] += 180  # Ajustar ângulos negativos
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q, r = 255, 255  # Pixels vizinhos
            
            # 0 graus (horizontal)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # 45 graus (diagonal descendente)
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i - 1, j + 1]
                r = magnitude[i + 1, j - 1]
            # 90 graus (vertical)
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i - 1, j]
                r = magnitude[i + 1, j]
            # 135 graus (diagonal ascendente)
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i + 1, j + 1]
                r = magnitude[i - 1, j - 1]
            
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
                
    return suppressed


#  LIMIARIZAÇÃO COM HISTERESIS t1 e t2

def threshold_hysteresis(image, low_ratio=0.3, high_ratio=0.02):
    high_threshold = np.max(image) * high_ratio
    low_threshold = high_threshold * low_ratio
    
    rows, cols = image.shape
    edges = np.zeros((rows, cols), dtype=np.uint8)

    strong = 255 # t1 
    weak = 85 # t2

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    edges[strong_i, strong_j] = strong
    edges[weak_i, weak_j] = weak

    return edges, strong, weak


#  RASTREAMENTO POR HISTERESIS

def edge_tracking_by_hysteresis(edges, strong, weak):
    rows, cols = edges.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if edges[i, j] == weak:
                if (edges[i-1:i+2, j-1:j+2] == strong).any():
                    edges[i, j] = strong
                else:
                    edges[i, j] = 0
    return edges


# EXECUTANDO CANNY


# Suavizar a imagem
blurred = gaussian_blur(img_gray)

# Calcular gradiente e direção
gradient_magnitude, gradient_direction = sobel_operator(blurred)

# Supressão de não-máximos
nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)

# Aplicar limiarização com histerese
thresholded_edges, strong, weak = threshold_hysteresis(nms_result)

# Aplicar rastreamento por histerese
final_edges = edge_tracking_by_hysteresis(thresholded_edges, strong, weak)


#======================================================================
#                     Canny
#====================================================================== 

#======================================================================
#                     Otsu
#======================================================================

def otsu_threshold(image):
    # Inicialize as variáveis, para n ocorrer overflow na formula
    weight_background = 0
    weight_foreground = 0
    mean_background = 0
    mean_foreground = 0
    weight_background = np.float64(weight_background)
    weight_foreground = np.float64(weight_foreground)
    mean_background = np.float64(mean_background)
    mean_foreground = np.float64(mean_foreground)


    hist = np.zeros(256, dtype=np.int32)  # Histograma para os niveis de cinza
    for pixel in image.ravel():  # Contagem de pixels
        hist[pixel] += 1
    
    total_pixels = image.size
    sum_total = np.sum(np.arange(256) * hist)  # Soma ponderada dos níveis de cinza
    sum_foreground, weight_foreground, weight_background = 0, 0, 0
    max_variance, threshold = 0, 0
    
    for t in range(256):
        weight_background += hist[t]  # numero de pixeis abaixo do limiar
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background  # numero de pixels acima do limiar
        if weight_foreground == 0:
            break
        
        sum_foreground += t * hist[t]  # Soma ponderada do primeiro plano
        mean_background = sum_foreground / weight_background  # Média do fundo
        mean_foreground = (sum_total - sum_foreground) / weight_foreground  # Média do primeiro plano
        
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t
    
    return threshold

def apply_threshold(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8) # Imagem binária
    binary_image[image >= threshold] = 255 # pixeis acima ou iguais ao limiar são brancos
    return binary_image

# Aplicar o método de Otsu
threshold_value = otsu_threshold(img_gray)
binary_otsu = apply_threshold(img_gray, threshold_value)

#======================================================================
#                     Otsu
#======================================================================

#======================================================================
#                    WaterShed
#======================================================================


#  marcadores watershed
def get_markers(image):
    markers = np.zeros_like(image, dtype=np.int32)
    markers[image < np.percentile(image, 50)] = 1  # Região de fundo
    markers[image > np.percentile(image, 50)] = 2  # Região de objeto
    return markers


#WaterShed
def watershed(image, markers):
    rows, cols = image.shape
    segmented = np.zeros_like(image, dtype=np.int32)
    
    # Inicializando as filas para cada marcador
    queue = []
    for i in range(rows):
        for j in range(cols):
            if markers[i, j] > 0:
                queue.append((i, j, markers[i, j]))
    
    while queue:
        i, j, label = queue.pop(0)
        if segmented[i, j] == 0:  # Se ainda não foi visitado
            segmented[i, j] = label
            
            # Vizinhos 4-conectados
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and segmented[ni, nj] == 0:
                    queue.append((ni, nj, label))
    
    return segmented

# Gerando os marcadores
markers = get_markers(img_gray)

# Aplicando Watershed
segmented_image = watershed(img_gray, markers)

#======================================================================
#                    WaterShed
#======================================================================



fig, ax = plt.subplots(1, 5, figsize=(10, 5))
ax[0].imshow(img_gray, cmap='gray')
ax[0].set_title("Imagem Original")
ax[0].axis("off")

ax[1].imshow(edges, cmap='gray')
ax[1].set_title("Bordas Marr-Hildreth")
ax[1].axis("off")

ax[2].imshow(final_edges, cmap='gray')
ax[2].set_title("Bordas Canny")
ax[2].axis("off")

ax[3].imshow(binary_otsu, cmap='gray')
ax[3].set_title(f"Binarização Otsu") # (Threshold={threshold_value})
ax[3].axis("off")

ax[4].imshow(segmented_image, cmap='jet')
ax[4].set_title("Segmentação Watershed")
ax[4].axis("off")



plt.show()



# comentario para responder a questão 2
# Os dois algoritmos aplicam um filtro gaussiano, mas o Marr-Hildreth usa derivada de laplace depois o a detecção de negativos
# já o Canny utiliza a função sobel para calcular gradientes e magnitude, depois a supressão de não-máximos e o histerese (t1 e t2)
# para delimitar as bordas fortes e fracas
# O Marr-Hidreth tem uma implementação mais simples mas é sensivel a rídos e gera bordas menos precisas e mais espessas
# O Canny tem uma implementação mais complexa, mas é mais preciso e gera contornos mais finos e bem definidos
# apesar disso precisa de um ajuste mais fino