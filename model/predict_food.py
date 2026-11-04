import sys
import os
import json
import warnings

# Suppress TensorFlow, CUDA, and oneDNN logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

def classify_image(image_path, top_k=5):
    """
    Classifies a food image using the pre-trained EfficientNetB0-Food101 model.
    Returns structured prediction results with top candidates sorted by confidence.
    """
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}
    
    try:
        import numpy as np
        from PIL import Image
        import tf_keras

        # Locate model files using path relative to current script directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'EfficientNetB0-Food101')
        config_path = os.path.join(model_dir, 'config.json')
        weights_path = os.path.join(model_dir, 'tf_model.h5')

        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            return {"success": False, "error": f"Model files missing in {model_dir}"}

        # Load Food-101 labels (101 classes)
        with open(config_path, 'r', encoding='utf-8') as f:
            id2label = json.load(f)['id2label']

        # Load EfficientNetB0 Food-101 Model
        model = tf_keras.models.load_model(weights_path, compile=False)

        # Load and resize image to 224x224 RGB using bilinear resampling
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        # Run inference
        raw_preds = model.predict(x, verbose=0)[0]

        # Sort indices by highest confidence first
        num_candidates = min(top_k, len(raw_preds))
        sorted_indices = np.argsort(raw_preds)[::-1][:num_candidates]

        top_label = id2label[str(sorted_indices[0])].title()
        top_prob = float(raw_preds[sorted_indices[0]])

        candidates = []
        for rank, idx in enumerate(sorted_indices, 1):
            candidates.append({
                "rank": rank,
                "label": id2label[str(idx)].title(),
                "confidence": round(float(raw_preds[idx]) * 100, 2)
            })

        return {
            "success": True,
            "image_path": image_path,
            "model_name": "EfficientNetB0-Food101",
            "prediction": top_label,
            "confidence_percent": round(top_prob * 100, 2),
            "candidates": candidates,
            "top3": candidates[:3]
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python predict_food.py <image_path>"}))
        sys.exit(1)

    img_arg = sys.argv[1]
    result = classify_image(img_arg)
    print(json.dumps(result, indent=2))
