import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.applications import InceptionV3, ResNet50, DenseNet121, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.image import resize

# Cargar CSV
train_df = pd.read_csv('datasets/sign_mnist_train.csv')
test_df = pd.read_csv('datasets/sign_mnist_test.csv')

# Separar labels y pixeles
y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Normalizar pixeles y convertir a imágenes 28x28
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Expandir a 3 canales (RGB)
X_train = np.repeat(X_train, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Convertir etiquetas a one-hot
num_classes = np.max(y_train) + 1
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


def resize_images(images, size):
    images_resized = tf.image.resize(images, size)
    return images_resized.numpy()

# Tamaño para los modelos (por ejemplo, 224x224)
img_size = 224

X_train_resized = resize_images(X_train, (img_size, img_size))
X_test_resized = resize_images(X_test, (img_size, img_size))


def build_model(base_model_class, input_shape=(img_size, img_size, 3), num_classes=num_classes):
    base_model = base_model_class(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Congelar pesos del base_model para transferencia
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = build_model(InceptionV3, input_shape=(img_size, img_size, 3), num_classes=num_classes)

history = model.fit(
    X_train_resized, y_train_cat,
    validation_data=(X_test_resized, y_test_cat),
    epochs=10,
    batch_size=32
)


