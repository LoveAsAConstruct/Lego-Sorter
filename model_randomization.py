# model_randomization.py
# A script used to generate random tensorflow models for testing
import tensorflow as tf
import random
from constants import HEIGHT, WIDTH, DEPTH

def generate_random_model(input_shape=(HEIGHT, WIDTH, DEPTH)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=input_shape))
    
    # Randomly decide the number of convolutional layers
    num_conv_layers = random.randint(1, 5)
    
    for _ in range(num_conv_layers):
        # Randomize parameters for each convolutional layer
        filters = 2 ** random.randint(0, 7)
        kernel_size = random.choice([3, 5])
        padding = random.choice(['same', 'valid'])
        model.add(tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, activation='relu'))
        
        # Randomly add a MaxPooling layer after each convolutional layer
        if random.random() < 0.5:
            pool_size = random.choice([2, 3])
            model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
    
    model.add(tf.keras.layers.Flatten())
    
    # Randomly add dropout layers
    if random.random() < 0.5:
        dropout_rate = random.random() * 0.5
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Randomly add dense layers
    num_dense_layers = random.randint(1, 3)
    for _ in range(num_dense_layers):
        units = random.choice([64, 128, 256])
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    
    # Add final layer
    model.add(tf.keras.layers.Dense(16))  # Adjust the number of output nodes if necessary
    
    return model