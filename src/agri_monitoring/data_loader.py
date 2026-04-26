import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, DATASET_PATH

def load_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH + "train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH + "val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ✅ SAVE class names BEFORE map
    class_names = train_ds.class_names

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
    val_ds   = val_ds.map(lambda x, y: (preprocess(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names   # ✅ return it