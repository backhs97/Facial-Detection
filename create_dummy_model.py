import tensorflow as tf
from tensorflow import keras

# Define a simple model
model = keras.Sequential([
    keras.layers.Input(shape=(48, 48, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save('emotion_model.h5')

print("Dummy emotion model created and saved as 'emotion_model.h5'")