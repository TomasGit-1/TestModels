import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionV3, ResNet50, DenseNet121, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# --- Cargar y preparar datos ---
train_df = pd.read_csv('datasets/sign_mnist_train.csv')
test_df = pd.read_csv('datasets/sign_mnist_test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(train_df['label'])

X_test = test_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_test = to_categorical(test_df['label'])

# Redimensionar y convertir a RGB 3 canales (esperado por modelos preentrenados)
def preprocess_images(X):
    X_resized = tf.image.resize_with_pad(X, 224, 224)
    X_rgb = tf.image.grayscale_to_rgb(X_resized)
    return X_rgb.numpy()

X_train_rgb = preprocess_images(X_train)
X_test_rgb = preprocess_images(X_test)

# --- Función para crear modelo ---
def build_model(base_model_class, input_shape=(224, 224, 3), num_classes=25):
    base_model = base_model_class(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Modelos a probar
models = {
    'InceptionV3': InceptionV3,
    'ResNet50': ResNet50,
    'DenseNet121': DenseNet121,
    'VGG16': VGG16
}

histories = {}
results = {}

# --- Entrenar y evaluar ---
for name, base_model_class in models.items():
    print(f"\n🚀 Entrenando {name}...")
    model = build_model(base_model_class)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(X_train_rgb, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=2)
    elapsed_time = time.time() - start_time
    
    test_loss, test_acc = model.evaluate(X_test_rgb, y_test, verbose=0)
    
    histories[name] = history
    results[name] = {
        'accuracy': test_acc,
        'loss': test_loss,
        'time': elapsed_time,
        'params': model.count_params()
    }
    
    print(f"✅ {name}: Accuracy={test_acc:.4f}, Tiempo={elapsed_time:.1f}s, Parámetros={model.count_params()}")

# --- Graficar resultados de precisión ---
plt.figure(figsize=(10,6))
for name, history in histories.items():
    plt.plot(history.history['val_accuracy'], label=name)
plt.title('Precisión en validación por epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# --- Imprimir tabla resumen ---
print("\nResumen de resultados:")
print(f"{'Modelo':<12}{'Accuracy':<10}{'Tiempo(s)':<10}{'Parámetros':<12}")
for name, res in results.items():
    print(f"{name:<12}{res['accuracy']:<10.4f}{res['time']:<10.1f}{res['params']:<12}")

