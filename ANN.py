#!/usr/bin/env python
# coding: utf-8
# @author: konain

import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing.preprocessing import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# pre-processing
data = preprocessing(_options="method_used_for_fraud")

X_train = data['train']['feature']
y_train = data['train']['label']

X_val = data['val']['feature']
y_val = data['val']['label']

X_test = data['test']['feature']
y_test = data['test']['label']

# Définition du modèle Fully Connected Layers avec BatchNorm, Dropout et Early Stopping
print("Entrainement du modèle")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(96, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(set(y_train)), activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping pour éviter l'overfitting
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',          # Monitors validation loss
    factor=0.5,                  # Reduces learning rate by half
    patience=5,                  # Number of epochs without improvement before reduction
    min_lr=1e-6,                 # Lower limit for learning rate
    verbose=1                    # Displays learning rate changes
)

# Entraînement du modèle
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, callbacks=[lr_scheduler])
print("Fin entrainement")

# Visualisation de la réduction de la fonction de coût
plt.figure(figsize=(6,5))
plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
plt.plot(history.epoch, history.history['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.title('Reduction of the cost function')
plt.legend()
plt.show()

# Évaluation du modèle
print("Evaluation du modèle")
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy val score : {accuracy}")

# Confusion matrix
plt.figure(figsize=(6,5))
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
