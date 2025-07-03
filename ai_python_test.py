import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape data if needed (e.g., for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)  # Add channel dimension
X_test = X_test.reshape(-1, 28, 28, 1)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer
    layers.MaxPooling2D((2, 2)),  # Pooling layer
    layers.Flatten(),  # Flatten the output
    layers.Dense(64, activation='relu'),  # Fully connected layer
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

predictions = model.predict(X_test)
predicted_classes = tf.argmax(predictions, axis=1)

# Example: Print the predicted class for the first test sample
print(f'Predicted class for the first test sample: {predicted_classes[0].numpy()}')
