import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data\\classification.csv')
print(data.columns)
file_paths = data['Filename'].values
labels = data['Classification'].values

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def load_and_preprocess_image(file_path, label):
    # Convert the file path tensor to a string
    file_path = file_path.numpy().decode('utf-8')
    # Join with base path
    full_path = os.path.join("data\\raw_dataset", file_path)

    # Load and preprocess the image
    img = tf.io.read_file(full_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [100, 100])
    img = tf.cast(img, tf.float32) / 255.0

    return img, label

def load_image(file_path, label):
    # Wrap the Python function in tf.py_function to allow it to operate on tensors
    [img, label] = tf.py_function(load_and_preprocess_image, [file_path, label], [tf.float32, label.dtype])
    # Set the shape manually because tf.py_function does not automatically set it
    img.set_shape([100, 100, 1])
    label.set_shape([])
    return img, label


train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

train_data = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_paths,test_labels))

train_data = train_data.map(load_image)
test_data = test_data.map(load_image)

train_data = train_data.batch(128)  
test_data = test_data.batch(128)

steps_per_epoch = len(train_paths) // 128  # Ensure you are calculating this based on actual data split sizes
validation_steps = len(test_paths) // 128  # Similar for validation data
print(f"taking {steps_per_epoch} steps from dataset of size {len(train_paths)}")
print(f"taking {validation_steps} steps from dataset of size {len(test_paths)}")

def print_dataset_shapes(dataset):
    for images, labels in dataset.take(1):  # Only take one batch to inspect
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)

print_dataset_shapes(train_data)
print_dataset_shapes(test_data)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 100, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(16, activation='softmax'),  # Assuming 16 classes
])


for images, labels in train_data.take(1):
    print("Batch loaded successfully")
    print("Images shape:", images.shape)
    print("Labels:", labels)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)
