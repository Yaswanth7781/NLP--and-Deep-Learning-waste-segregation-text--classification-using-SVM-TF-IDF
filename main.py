import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import joblib
import warnings
import sys
from google.colab import files # Import Colab's file tool

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- 1. DEFINE PATHS AND CONSTANTS ---
CNN_MODEL_PATH = '/content/waste_cnn_model.keras'
SVM_MODEL_PATH = '/content/svm_text_pipeline.joblib'
IMG_SIZE = (160, 160) # Must match your training IMG_SIZE

# ⚠️ UPDATE THIS: Path to your original merged dataset folder
MERGED_DATA_DIR = "/content/merged_dataset" 

# --- 2. GLOBAL VARIABLES FOR MODELS ---
loaded_cnn_model = None
loaded_text_pipeline = None
loaded_class_names = []

# --- 3. PREDICTION FUNCTION ---
def predict_waste_from_image(image_path):
    """
    Takes a path to an image, runs it through the two-stage
    pipeline, and returns the item name and its broad category.
    """
    try:
        # --- Stage 1: CNN Prediction ---
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = loaded_cnn_model.predict(img_array, verbose=0) 
        score = tf.nn.softmax(predictions[0])
        
        predicted_index = np.argmax(score)
        predicted_item_name = loaded_class_names[predicted_index]
        confidence = 100 * np.max(score)

        # --- Stage 2: SVM Text Prediction ---
        final_category = loaded_text_pipeline.predict([predicted_item_name])[0]
        
        return {
            "file": os.path.basename(image_path),
            "predicted_item": predicted_item_name,
            "confidence": f"{confidence:.2f}%",
            "final_category": final_category
        }
    except Exception as e:
        return {
            "file": os.path.basename(image_path),
            "error": f"Could not process image. Error: {e}"
        }

# --- 4. MAIN PROGRAM EXECUTION ---
def main():
    global loaded_cnn_model, loaded_text_pipeline, loaded_class_names
    
    try:
        # --- Load Models and Class Names ---
        print("Loading models, please wait...")

        # A. Get Class Names
        if not os.path.exists(MERGED_DATA_DIR):
            print(f"Error: Dataset directory not found at {MERGED_DATA_DIR}.")
            return
        loaded_class_names = sorted([d for d in os.listdir(MERGED_DATA_DIR) 
                                     if os.path.isdir(os.path.join(MERGED_DATA_DIR, d))])
        if len(loaded_class_names) == 0:
            print(f"Error: No class subfolders found in {MERGED_DATA_DIR}.")
            return
        print(f"Found {len(loaded_class_names)} classes.")

        # B. Load CNN Model
        if not os.path.exists(CNN_MODEL_PATH):
            print(f"Error: CNN model not found at {CNN_MODEL_PATH}")
            return
        loaded_cnn_model = load_model(CNN_MODEL_PATH)
    
        # C. Load SVM Model
        if not os.path.exists(SVM_MODEL_PATH):
            print(f"Error: SVM model not found at {SVM_MODEL_PATH}")
            return
        loaded_text_pipeline = joblib.load(SVM_MODEL_PATH)
    
        print("✅ Models loaded successfully!\n")

        # --- Ask for user to upload files ---
        print("Please choose one or more images to classify:")
        
        # This line opens the file upload dialog
        uploaded = files.upload() 

        if not uploaded:
            print("No files selected. Exiting.")
            return

        print(f"\nFound {len(uploaded)} file(s). Starting classification...")
        
        # --- Process each uploaded file ---
        for i, (filename, content) in enumerate(uploaded.items()):
            print(f"\n--- [Image {i+1}/{len(uploaded)}] ---")
            
            # The uploaded file is in memory (content)
            # We must write it to a temporary file for our function to read
            temp_path = os.path.join("/tmp", filename)
            with open(temp_path, 'wb') as f:
                f.write(content)

            # Run the prediction on the temporary file
            result = predict_waste_from_image(temp_path)
            
            # Print the result
            if "error" in result:
                print(f"File:  {result['file']}")
                print(f"Error: {result['error']}")
            else:
                print(f"File:           {result['file']}")
                print(f"Predicted Item: {result['predicted_item']}")
                print(f"Confidence:     {result['confidence']}")
                print(f"Final Category: {result['final_category']}")
            
            # Clean up the temporary file
            os.remove(temp_path)

    except Exception as e:
        print(f"\n--- A fatal error occurred ---")
        print(e)

# Run the main function
if __name__ == "__main__":
    main()