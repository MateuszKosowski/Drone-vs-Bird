
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

path_to_data = "dataset"
batch_size = 16 
img_size = (224, 224)
RESULTS_DIR = "results"


# Display TensorFlow version and available GPU devices
print("Wersja TensorFlow: ", tf.__version__)
print("Dostępne urządzenia GPU: ", tf.config.list_physical_devices('GPU'))

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Initialize summary metrics file
metrics_file = os.path.join(RESULTS_DIR, "summary_metrics.csv")
with open(metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Train_Size', 'Test_Size', 'Accuracy', 'Precision', 'Recall', 'F1_Score'])


def evaluate_split(test_size):

    # Calculate train/test percentages
    train_percent = int(round((1 - test_size) * 100))
    test_percent = int(round(test_size * 100))
    split_name = f"{train_percent}_{test_percent}"

    print(f"\n{'='*50}\nEwaluacja: Trening {train_percent}% / Test {test_percent}%\n{'='*50}")

    
    # Method to create train dataset - couple: image and its label
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_data, # Label will be assigned based on the directory name
        validation_split=test_size, # Presentage of data to reserve for validation
        subset="training", # Use the training subset
        seed=123, # Seed for reproducibility
        image_size=img_size, # Resize images to a consistent size (224x224)
        batch_size=batch_size, # Number of images to process in 1 step in epoch
        crop_to_aspect_ratio=True # 1. Scales the photo so that the shorter side fits the target size 2. Crops the longer side to fit the target size
    )

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

    # Augmentation
    # It is a technique to artificially increase the size of the training dataset by creating modified versions of images in the dataset. 
    # This helps improve the model's ability to generalize and reduces overfitting. 
    # In each epoch, the model will see a different version of the image, which helps it learn more robust features and patterns in the data.
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), # Left-right flip - mirror effect
        tf.keras.layers.RandomRotation(0.1), # 10% rotation so it means +- 36 degrees
    ])

    # Build the CNN model
    model = tf.keras.Sequential([

        # Input layer to specify the shape of the input images (224x224 pixels with 3 color channels)
        tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3)),

        data_augmentation,

        # Rescaling layer to normalize pixel values from [0, 255] to [0, 1]
        tf.keras.layers.Rescaling(1./255),
        
        # Set of 32 filters (3x3) to extract simple features like edges. Example filters: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] detects horizontal edges
        # Relu function: f(x) = max(0, x) - if the input is negative, it outputs 0; if the input is positive, it outputs the input value.
        # Padding 'same' means to add zeros around the input image so that the output has the same size as the original input.
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),

        # Example of output of Conv2D layer with 4 filters (3x3) and padding 'same' for a 4x4 input image:
        # [ 1,  3,  2,  1 ]
        # [ 4,  1,  1,  0 ]
        # [ 0,  2,  5,  2 ]
        # [ 1,  1,  3,  4 ]
        # Max pooling is used to reduce the spatial dimensions of the feature maps.
        # It works by sliding a window (in this case, 2x2) across the input feature map and taking the maximum value within that window.
        # Thanks to this, we highlight the fact that a given pattern appeared in a given area, regardless of whether it was 5px next to it
        tf.keras.layers.MaxPooling2D(),
        
        # Another convolutional layer with 64 filters to capture more complex features
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        
        # Final convolutional layer with 128 filters to capture even more complex features
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        
        # To convert 3D feature maps into a 1D vector
        tf.keras.layers.Flatten(),
        
        # Classic neural layer. 128 neurons "analyze" features extracted by previous layers
        tf.keras.layers.Dense(128, activation='relu'),

        # Dropout is a techniques to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time, which helps the model to generalize better.
        tf.keras.layers.Dropout(0.3), 

        # Last layer with softmax activation to output probabilities for each class. The number of neurons is equal to the number of classes in the dataset.
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    # Adam is an optimization algorithm that adjusts the learning rate during training, which helps the model converge faster and more efficiently.
    # Sparse categorical crossentropy is a loss function used for multi-class classification problems where the labels are provided as integers. 
    # It measures the difference between the predicted probabilities and the true labels, encouraging the model to output high probabilities for the correct class and low probabilities for the incorrect classes.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\n" + "-"*20 + " ROZPOCZĘCIE TRENINGU " + "-"*20)

    # Early stopping is a technique to prevent overfitting by monitoring the validation loss during training and stopping the training process if the validation loss does not improve for a specified number of epochs (patience).
    callback = tf.keras.callbacks.EarlyStopping (
        monitor='val_loss',  # It could be 'val_accuracy' if we want to monitor accuracy instead of loss
        patience=5,         
        restore_best_weights=True 
    )

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=100, 
        callbacks=[callback] 
    )
    
    # Save training history to CSV for later analysis
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(RESULTS_DIR, f"training_history_{split_name}.csv"), index_label="epoch")

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

    # Saving confusion matrix to CSV for later analysis
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(RESULTS_DIR, f"confusion_matrix_{split_name}.csv"))

    # Append summary metrics to the CSV file
    with open(metrics_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"{train_percent}%", f"{test_percent}%", acc, prec, rec, f1])

test_splits_to_evaluate = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] 

for split in test_splits_to_evaluate:
    evaluate_split(split)