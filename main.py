import tensorflow as tf

print("Wersja TensorFlow: ", tf.__version__)
print("Dostępne urządzenia GPU: ", tf.config.list_physical_devices('GPU'))

path_to_data = "dataset"
batch_size = 16 
img_size = (384, 384)  
test_size = 0.2

print("Ładowanie danych treningowych...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_to_data,
    validation_split=test_size,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    crop_to_aspect_ratio=True
)

print("Ładowanie danych walidacyjnych...")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_to_data,
    validation_split=test_size,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    crop_to_aspect_ratio=True
)

class_names = train_ds.class_names
print("Klasy: ", class_names)