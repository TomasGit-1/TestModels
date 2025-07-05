import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50, DenseNet121
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_incep
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

# ---------------------- CONFIGURACIÓN ----------------------
N_FRAMES = 4         # Frames por secuencia
EPOCHS = 5
BATCH_SIZE = 32

# ---------------------- MODELOS CNN + RNN ----------------------
models_cnn = {
    'InceptionV3': (InceptionV3, preprocess_incep),
    'ResNet50': (ResNet50, preprocess_resnet),
    'DenseNet121': (DenseNet121, preprocess_dense)
}

models_rnn = ['LSTM', 'GRU']

# ---------------------- CARGA Y FORMATO ----------------------
df = pd.read_csv('datasets/sign_mnist_train.csv')
X = df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.
y = to_categorical(df['label'])
num_classes = y.shape[1]

# Crear secuencias
total_samples = len(X) // N_FRAMES
X = X[:total_samples * N_FRAMES]
y = y[:total_samples * N_FRAMES:N_FRAMES]
X_seq = X.reshape((total_samples, N_FRAMES, 28, 28, 1))

# ---------------------- PREPROCESAMIENTO RGB ----------------------
def preprocess_sequence(seq, target_size=(224, 224)):
    rgb_seq = []
    for frame in seq:
        resized = tf.image.resize_with_pad(frame, target_size[0], target_size[1])
        rgb = tf.image.grayscale_to_rgb(resized)
        rgb_seq.append(rgb)
    return np.array(rgb_seq)

# Preprocesar todas las secuencias
X_seq_rgb = np.array([preprocess_sequence(seq) for seq in X_seq])

# ---------------------- RESULTADOS ----------------------
results = []

for cnn_name, (cnn_class, preprocess_fn) in models_cnn.items():
    print(f"\n📦 Extrayendo embeddings con {cnn_name}...")
    base_model = cnn_class(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # Extraer embeddings (N_videos, N_FRAMES, D)
    X_embeddings = []
    for video in X_seq_rgb:
        frames = preprocess_fn(video)
        feats = base_model.predict(frames, verbose=0)
        X_embeddings.append(feats)
    X_embeddings = np.array(X_embeddings)

    for rnn_type in models_rnn:
        print(f"\n🚀 Entrenando {cnn_name} + {rnn_type}...")
        X_train, X_val, y_train, y_val = train_test_split(X_embeddings, y, test_size=0.2)

        model = Sequential()
        if rnn_type == 'LSTM':
            model.add(LSTM(128, input_shape=(N_FRAMES, X_embeddings.shape[-1])))
        else:
            model.add(GRU(128, input_shape=(N_FRAMES, X_embeddings.shape[-1])))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        elapsed = time.time() - start_time

        final_acc = history.history['val_accuracy'][-1]
        results.append({
            'Modelo': f'{cnn_name}+{rnn_type}',
            'Accuracy': round(final_acc, 4),
            'Tiempo(s)': round(elapsed, 1),
            'Parámetros': model.count_params()
        })
        print(f"✅ {cnn_name}+{rnn_type} - Acc: {final_acc:.4f} - Tiempo: {elapsed:.1f}s - Paráms: {model.count_params()}")

# ---------------------- MOSTRAR RESULTADOS ----------------------
print("\n📊 Resultados resumen:")
print(f"{'Modelo':<25}{'Accuracy':<10}{'Tiempo(s)':<10}{'Parámetros'}")
for r in results:
    print(f"{r['Modelo']:<25}{r['Accuracy']:<10}{r['Tiempo(s)']:<10}{r['Parámetros']}")
