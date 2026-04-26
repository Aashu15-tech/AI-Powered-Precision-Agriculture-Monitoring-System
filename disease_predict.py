# =============================================================================
#  disease_predict.py — Predict crop disease from photo using model.keras
#
#  Input  : crop photo (PIL Image or file path)
#  Output : {'label': 'Disease'/'No Disease', 'confidence': 87.3, 'raw_prob': 0.873}
# =============================================================================

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config.config import IMG_SIZE, CLASS_NAMES


# =============================================================================
# LOAD KERAS MODEL
# =============================================================================

_disease_model = None   # module-level cache — load only once


def load_disease_model(model_path: str = 'models/model.keras'):
    """
    Load model.keras from disk. Cached after first load.

    Args:
        model_path (str): Path to model.keras file.

    Returns:
        keras Model object.
    """
    global _disease_model
    if _disease_model is not None:
        return _disease_model

    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[✗] model.keras not found at '{model_path}'")

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import BatchNormalization

    # Subclass BatchNormalization to silently drop unsupported args
    class CompatBatchNorm(BatchNormalization):
        def __init__(self, **kwargs):
            kwargs.pop('renorm', None)
            kwargs.pop('renorm_clipping', None)
            kwargs.pop('renorm_momentum', None)
            super().__init__(**kwargs)

    _disease_model = load_model(
        model_path,
        custom_objects={'BatchNormalization': CompatBatchNorm}
    )
    print(f"[✓] Disease model loaded → {model_path}")
    return _disease_model   
#     global _disease_model

#     if _disease_model is not None:
#         return _disease_model

#     import os
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(
#             f"[✗] model.keras not found at '{model_path}'\n"
#             f"    Sahi path config mein set karo."
#         )

#     # Lazy import — tensorflow heavy, load only when needed
#     from tensorflow.keras.models import load_model
#     _disease_model = load_model(model_path)
#     print(f"[✓] Disease model loaded → {model_path}")
#     print(f"    Input shape expected : {INPUT_SHAPE}")
#     return _disease_model


# # =============================================================================
# # PREPROCESS UPLOADED PHOTO
# # =============================================================================

def preprocess_image(image_input) -> np.ndarray:
    """
    Resize and normalize crop photo for model.keras inference.

    Reads IMG_SIZE from config.py automatically.

    Args:
        image_input: One of —
            - PIL.Image object (from st.file_uploader)
            - str file path
            - bytes (raw image bytes)

    Returns:
        np.ndarray: Shape (1, IMG_SIZE, IMG_SIZE, 3), normalized to [0, 1]
    """
    from PIL import Image
    import io

    # Handle different input types
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        # PIL Image or file-like object (st.file_uploader returns BytesIO)
        img = Image.open(image_input).convert('RGB')

    # Resize to model's expected input size (from config)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy + normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Add batch dimension → (1, IMG_SIZE, IMG_SIZE, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr


# =============================================================================
# PREDICT DISEASE
# =============================================================================

def predict_disease(image_input,
                    model_path: str = 'models/model.keras') -> dict:
    """
    Full disease prediction pipeline for one crop photo.

    Args:
        image_input : PIL Image / file path / bytes / BytesIO
        model_path  : Path to model.keras

    Returns:
        dict: {
            'label'      : predicted class name from CLASS_NAMES,
            'confidence' : float (e.g. 87.3 → 87.3%),
            'raw_probs'  : dict of {class_name: probability%} for all classes,
        }
    """
    model = load_disease_model(model_path)
    img   = preprocess_image(image_input)

    # Model output: softmax → shape (1, 3)
    raw_output = model.predict(img, verbose=0)
    probs = raw_output[0]  # shape: (3,) — one probability per class

    # Get predicted class
    predicted_idx = int(np.argmax(probs))
    label         = CLASS_NAMES[predicted_idx]
    confidence    = round(float(np.max(probs)) * 100, 1)

    return {
        'label'      : label,
        'confidence' : confidence,
        'raw_probs'  : {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1)
                       for i in range(len(CLASS_NAMES))}
    }