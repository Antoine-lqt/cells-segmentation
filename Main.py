from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import erosion, opening
import numpy as np
import cv2

file_path = "chemin d'accès du fichier"

# Ouvrir le fichier TIF
tif_file = Image.open(file_path)

# Créez une liste pour stocker les images
images_list = []

try:
    while True:
        # Accéder à l'image spécifique (par son numéro d'index)
        tif_file.seek(len(images_list))

        # Lire l'image
        image = tif_file.copy()

        # Ajouter l'image à la liste
        images_list.append(image)

except EOFError:
    print("Toutes les images ont été chargées dans la liste.")

# Fermer le fichier TIF
tif_file.close()


def seuillage(image, seuil):
    image_binaire = image.point(lambda p: p > seuil and 255)

    return image_binaire


def calculate_and_round(values_list, number):
    total_added_values = 0
    for value in values_list:
        ratio = value / number
        if ratio >= 2:
            total_added_values += int(ratio) - 1
    total_number=len(values_list)+total_added_values
    return total_number

element_structurant = np.array([[0, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 0]], dtype=bool)
nombre_ROI=[]
mediane_ROI=[]
nombre_ROI_avant_ratio=[]
seuil=70

for i in range(len(images_list)):
    image = images_list[i]

    image_grayscale = image.convert("L")

    min_gray = min(image_grayscale.getdata())
    max_gray = max(image_grayscale.getdata())

    # Normaliser l'image en ajustant les valeurs des niveaux de gris
    normalized_image = Image.eval(image_grayscale, lambda x: int(255 * (x - min_gray) / (max_gray - min_gray)))

    seuil_image= np.array(seuillage(normalized_image,seuil))

    eroded_image= erosion(seuil_image,element_structurant)

    erod_open_image = opening(eroded_image, element_structurant)
    
# Appliquer Watershed sur l'image en niveaux de gris
    _, markers = cv2.connectedComponents(erod_open_image)

# Créer une image en couleur vide avec les mêmes dimensions
    erod_open_image_color = cv2.cvtColor(erod_open_image, cv2.COLOR_GRAY2BGR)

# Appliquer Watershed
    cv2.watershed(erod_open_image_color, markers)

# Colorer les régions inconnues en bleu
    erod_open_image_color[markers == -1] = [0, 0, 255]

    num_cells = len(np.unique(markers)) - 1  # Soustrayez 1 pour exclure les régions inconnues (-1)

    unique_markers, marker_sizes = np.unique(markers, return_counts=True)

# Exclure la région inconnue
    unique_markers = unique_markers[1:]
    marker_sizes = marker_sizes[1:]

# Liste des tailles des marqueurs
    marker_sizes_list = marker_sizes.tolist()
    marker_sizes_list = [size for size in marker_sizes_list if size < 10000]
    

    median_size = np.median(marker_sizes_list)

# Compter le nombre de cellules marquées
    num_cells = len(np.unique(markers)) - 1  # Soustrayez 1 pour exclure les régions inconnues (-1)

# Générer un tableau d'indices de marqueurs
    marker_indices = np.arange(1, num_cells + 1)  # Commence à 1 car 0 est la région inconnue
    
    num_regions_of_interest=calculate_and_round(marker_sizes_list, median_size)
    
    print(f"Nombre de régions d'intérêt : {len(marker_sizes_list)}")      
    print(f"Médiane des superficies : {median_size}")
    print(f"Nombre de régions d'intérêt (après ajustement du ratio) : {num_regions_of_interest}")
    
    
    
    nombre_ROI_avant_ratio.append(len(marker_sizes_list))
    nombre_ROI.append(num_regions_of_interest)
    mediane_ROI.append(median_size)


        # Tracer les contours des régions identifiées
    contours, _ = cv2.findContours(erod_open_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une copie de l'image en couleur pour l'affichage
    marked_image = cv2.cvtColor(erod_open_image.copy(), cv2.COLOR_GRAY2BGR)

    # Dessiner les contours sur l'image
    cv2.drawContours(marked_image, contours, -1, (0, 255, 0), 2)  # -1 pour dessiner tous les contours

    # Afficher l'image avec les contours
    plt.imshow(marked_image)
    plt.title(f"Image {i+1} avec les contours des cellules identifiées")
    plt.show()
    
    #plt.figure()
    #plt.imshow(erod_open_image_color)
    #plt.show()
    

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Afficher l'image avec les contours des cellules identifiées
ax1.imshow(marked_image, cmap='gray')
ax1.set_title('Position des marqueurs identifiés')
ax1.set_xlabel('Pixels')
ax1.set_ylabel('Pixels')

# Afficher l'image dont le niveau de gris a été balancé
ax2.imshow(normalized_image, cmap='viridis')
ax2.set_title('Image dont le niveau de gris a été balancé')
ax2.set_xlabel('Pixels')
ax2.set_ylabel('Pixels')

# Lien de vue entre les deux sous-graphiques
def on_xlim_change(ax):
    ax2.set_xlim(ax.get_xlim())

def on_ylim_change(ax):
    ax2.set_ylim(ax.get_ylim())

ax1.callbacks.connect('xlim_changed', on_xlim_change)
ax1.callbacks.connect('ylim_changed', on_ylim_change)

# Titre global
fig.suptitle('Coupe de l\'image 16')

plt.show()







