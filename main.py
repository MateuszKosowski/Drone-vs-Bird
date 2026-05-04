
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("Wersja TensorFlow: ", tf.__version__)
print("Dostępne urządzenia GPU: ", tf.config.list_physical_devices('GPU'))

path_to_data = "dataset"
batch_size = 16 
img_size = (224, 224)
RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR) 

def evaluate_split(test_size):
    print(f"\n{'='*50}\nEwaluacja: Trening {100-int(test_size*100)}% / Test {int(test_size*100)}%\n{'='*50}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_data, validation_split=test_size, subset="training", seed=123,
        image_size=img_size, batch_size=batch_size, crop_to_aspect_ratio=True
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_data, validation_split=test_size, subset="validation", seed=123,
        image_size=img_size, batch_size=batch_size, crop_to_aspect_ratio=True
    )

    class_names = train_ds.class_names

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\n" + "-"*20 + " ROZPOCZĘCIE TRENINGU " + "-"*20)

    callback = tf.keras.callbacks.EarlyStopping (
        monitor='val_loss', 
        patience=5,         
        restore_best_weights=True 
    )

    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=100, 
        callbacks=[callback] 
    )

    print("\n" + "-"*20 + " ROZPOCZĘCIE TESTOWANIA " + "-"*20)
    print("Generowanie predykcji...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- WYNIKI I MIARY JAKOŚCI ---")
    print(f"1. Dokładność (Accuracy): {acc:.4f}")
    print(f"2. Precyzja (Precision):  {prec:.4f}")
    print(f"3. Czułość (Recall):      {rec:.4f}")
    print(f"4. F1 Score:              {f1:.4f}")
    print("\n5. Macierz pomyłek (Confusion Matrix):")
    print(cm)
    
test_splits_to_evaluate = [0.2, 0.3] 

for split in test_splits_to_evaluate:
    evaluate_split(split)