import tensorflow as tf
import numpy as np
from config import IMG_SIZE, MODEL_PATH
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

model = tf.keras.models.load_model(MODEL_PATH)

import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224)
)

class_names = train_ds.class_names
def predict(image_path, class_names):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)   # ✅ FIX

    prediction = model.predict(img_array, verbose=0)

    print("Raw prediction:", prediction)

    class_id = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))

    return class_names[class_id], confidence

if __name__ == "__main__":

    label, conf = predict(r"C:\disease prediction\dataset\val\Potato___Early_blight\agri1.webp", class_names)

    print("Disease:", label)
    print("Confidence: " , conf)

# def predict(image_path, class_names):
#     img = tf.keras.preprocessing.image.load_img(
#         image_path, target_size=(IMG_SIZE, IMG_SIZE)
#     )
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     prediction = model.predict(img_array)
#     class_id = np.argmax(prediction)
#     confidence = np.max(prediction)

#     return class_names[class_id], confidence