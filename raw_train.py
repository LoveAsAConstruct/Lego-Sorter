import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from contextlib import redirect_stdout
from datetime import datetime

HEIGHT = 180
WIDTH = 180

def run_training_and_save_all(data_dir, model, base_run_dir='train_results/t', epochs=10, batch_size=32):
    # Setup
    AUTOTUNE = tf.data.AUTOTUNE

    # Load the data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=batch_size
    )

    # Access class names and determine the number of classes
    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Configure datasets
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Generate a unique directory name with the current date and time
    current_time = datetime.now().strftime("%d_%H-%M-%S")
    run_dir = f"{base_run_dir}_{current_time}_epochs-{epochs}"
    os.makedirs(run_dir, exist_ok=True)

    # Save model and weights
    model_path = os.path.join(run_dir, 'model')
    weights_path = os.path.join(run_dir, 'weights')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    model.save(os.path.join(model_path, 'model.h5'))
    model.save_weights(os.path.join(weights_path, 'weights.h5'))

    # Save summary
    with open(os.path.join(model_path, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history_df.index + 1
    results_path = os.path.join(run_dir, 'results')
    os.makedirs(results_path, exist_ok=True)
    history_df.to_csv(os.path.join(results_path, 'training_history.csv'), index=False)

    # Generate and save plots
    epochs_range = range(epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(results_path, 'results_graph.png'))
    plt.close()


model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(HEIGHT, WIDTH, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(16)
])
run_training_and_save_all("archive\LEGO brick images v1", model)