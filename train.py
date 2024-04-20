import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from contextlib import redirect_stdout
from datetime import datetime
from constants import *

def run_training_and_save_all(model, base_run_dir='train_results', epochs=10, batch_size=32, data_dir="archive\LEGO brick images v1", verbose=False, name='train'):
    # Function to evaluate the effectiveness of a function (relatively ambiguous)
    def model_effectiveness(loss, accuracy, duration):
        correctness = (1 - loss) * accuracy
        return correctness * 100 / (duration ** 0.25)
    
    AUTOTUNE = tf.data.AUTOTUNE

    # Load validation and training data
    training_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=batch_size
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        batch_size=batch_size
    )

    if verbose:
        class_names = training_data.class_names
        num_training_samples = sum(1 for _ in training_data)
        print(f"Training for {len(class_names)} class(es): {class_names}")
        print(f"Loaded {len(validation_data)} validation datapoints")
        print(f"Loaded {num_training_samples} training datapoins")

    # Configure data
    training_data = training_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    start_time = time.time()

    # Train the model
    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs
    )

    # Calculate training duration
    end_time = time.time()
    training_time = end_time - start_time

    loss, accuracy = model.evaluate(validation_data)
    effectiveness = model_effectiveness(loss,accuracy,training_time)
    report = f'Loss: {loss}, Accuracy: {accuracy}, Duration: {training_time} seconds, Effectiveness: {model_effectiveness(loss,accuracy,training_time)}'
    print(report)

    if base_run_dir == None:
        # Do nothing if no datapath
        return
    elif os.path.exists(base_run_dir):
        # Save run data if datapath exists
        # Generate the run directory
        current_time = datetime.now().strftime("%d_%H-%M-%S")
        run_dir = f"{base_run_dir}/{name}_{current_time}_epochs-{epochs}_loss-{round(loss,2)}_accuracy-{round(accuracy,2)}_E-{round(effectiveness,2)}"
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

        with open(os.path.join(results_path, 'loss_accuracy.txt'), 'w') as f:
            with redirect_stdout(f):
                print(report)

        # Generate plots
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

        if verbose:
            print(f"Results saved to {run_dir}")
    else:
        # Raise an error if path not found
        raise FileNotFoundError(f"The base save directory '{base_run_dir}' does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save lego piece detection models")

    # Commandline arguments
    parser.add_argument("--random", action="store_true", help="Will generate a random model to train, otherwise uses ol' reliable")
    parser.add_argument("--epochs", type=int, default=10, help="# of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="archive\LEGO brick images v1", help="Directory containing lego images")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--save_dir", type=str, default="train_results", help="Directory to save runtime folder")
    parser.add_argument("--runs", type=int, default=1, help="# of runs to test (reccomended with random)")

    args = parser.parse_args()
    for _ in range(0,args.runs):
        # Select model
        if args.random:
            from model_randomization import generate_random_model
            model = generate_random_model()
        else:
            # ol' reliable
            model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255, input_shape=(HEIGHT, WIDTH, 3)),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(16)
            ])

        # Run training
        run_training_and_save_all(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            verbose=args.verbose,
            name=args.name,
            base_run_dir=args.save_dir
        )
