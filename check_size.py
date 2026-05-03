import os
from PIL import Image

def analyze_image_sizes(dataset_path):
    min_width = float('inf')
    min_height = float('inf')
    max_width = 0
    max_height = 0
    
    min_img = None
    max_img = None
    
    total_images = 0
    invalid_files = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    total_images += 1
                    
                    if width * height < min_width * min_height:
                        min_width, min_height = width, height
                        min_img = file_path
                        
                    if width * height > max_width * max_height:
                        max_width, max_height = width, height
                        max_img = file_path

            except Exception as e:
                invalid_files += 1

    print(f"Ilość przeanalizowanych zdjęć: {total_images}")
    if invalid_files > 0:
         print(f"Pominięto błędnych plików: {invalid_files}")
         
    if total_images > 0:
        print("\n--- Najmniejsze zdjęcie ---")
        print(f"Rozmiar: {min_width} x {min_height} pikseli")
        print(f"Plik: {min_img}")
        
        print("\n--- Największe zdjęcie ---")
        print(f"Rozmiar: {max_width} x {max_height} pikseli")
        print(f"Plik: {max_img}")
    else:
        print("Nie znaleziono żadnych zdjęć w podanym folderze.")

if __name__ == "__main__":
    analyze_image_sizes("dataset")