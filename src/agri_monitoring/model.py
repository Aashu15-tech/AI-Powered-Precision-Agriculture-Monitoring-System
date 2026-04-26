import tensorflow as tf
from keras import layers, models
from config import IMG_SIZE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.4),
    layers.RandomContrast(0.4),
    layers.RandomBrightness(0.3),
])

def build_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        data_augmentation,

        # OPTIONAL (only if IMG_SIZE > 224)
        # layers.CenterCrop(224, 224),

        layers.GaussianNoise(0.05),

        base_model,

        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model



# import tensorflow as tf
# from keras import layers, models
# from config import IMG_SIZE

# def build_model(num_classes):
#     base_model = tf.keras.applications.EfficientNetB0(
#         input_shape=(IMG_SIZE, IMG_SIZE, 3),
#         include_top=False,
#         weights='imagenet'
#     )

#     base_model.trainable = True

#     model = models.Sequential([
#         layers.Rescaling(1./255),
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.BatchNormalization(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(num_classes, activation='softmax')
#     ])

#     return model