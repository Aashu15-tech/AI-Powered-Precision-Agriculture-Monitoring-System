from src.data_loader import load_data
from src.model import build_model
from config import EPOCHS, MODEL_PATH
import tensorflow as tf

# Load data
train_ds, val_ds, class_names = load_data()

# Build model
model = build_model(len(class_names))

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True
    )
]

# Train (NO class_weight)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save model (use new format)
model.save(MODEL_PATH)


# import tensorflow as tf

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "dataset/train",
#     image_size=(224, 224)
# )

# print(train_ds.class_names)

# from src.data_loader import load_data
# from src.model import build_model
# from config import EPOCHS, MODEL_PATH
# import tensorflow as tf

# train_ds, val_ds = load_data()

# model = build_model(len(train_ds.class_names))

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# model.save(MODEL_PATH)