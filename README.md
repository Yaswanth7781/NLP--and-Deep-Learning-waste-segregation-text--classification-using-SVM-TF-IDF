# NLP--and-Deep-Learning-waste-segregation-text--classification-using-SVM-TF-IDF
An intelligent waste sorter using a two-stage AI model: a CNN identifies specific items from images (e.g., "apple", "cardboard"), and an SVM classifies those labels into broad categories (e.g., "Organic Waste", "Recyclable Paper"), automating waste segregation for proper disposal.
Smart Waste Sorter: A Two-Stage AI Classification System
This project is an intelligent waste sorter that automatically classifies waste from an image into its proper disposal category.

💡 Project Overview
The system uses a two-stage hybrid AI model to achieve this:

Goal: To automatically classify waste from an image into broad disposal categories (e.g., "Organic Waste," "Recyclable Paper").

Stage 1 (Vision): A Convolutional Neural Network (CNN), built with TensorFlow/Keras, recognizes the specific item in an image (e.g., "banana peel," "cardboard").

Stage 2 (Classification): An SVM (Support Vector Machine) model, using Scikit-learn and TF-IDF, takes the text label from the CNN and classifies it into a final, broad category.

⚙️ How It Works
This cascaded pipeline leverages the strengths of both models:

A user provides an image of a waste item.

The CNN Model (e.g., waste_cnn_model.keras) analyzes the image pixels and predicts a specific class label (e.g., the text string "apple").

This text label is then fed into the SVM Model (e.g., svm_text_pipeline.joblib).

The SVM, which has been trained on a text map, classifies the string "apple" into its pre-defined broad category: "Organic Waste".

The final category is presented to the user.

Flow: `

Shutterstock

→CNN Model→"apple"(text) →SVM Model→"Organic Waste"` (final category)

🚀 Getting Started
Follow these steps to set up and run the project.

1. Dependencies
You'll need Python 3.x and the following libraries. You can install them all using pip:

Bash

pip install tensorflow scikit-learn numpy joblib pillow
You can also place this in a requirements.txt file and run pip install -r requirements.txt.

2. Dataset Structure
This project requires a specific dataset folder structure for training. The CNN model (image_dataset_from_directory) learns the class names directly from the subfolder names.

Your main dataset folder (e.g., merged_dataset/) must look like this:

/merged_dataset/
├── apple/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── banana/
│   ├── img_001.jpg
│   └── ...
├── cardboard/
│   ├── img_001.jpg
│   └── ...
├── metal/
│   ├── img_001.jpg
│   └── ...
├── trash/
│   ├── img_001.jpg
│   └── ...
└── (and 21+ other class folders)
🏁 How to Run
The project is split into three main scripts.

1. Train the CNN Model (Stage 1)
This script trains the image recognition model. It will read from your merged_dataset folder, train a new CNN (using transfer learning from MobileNetV2), and save the result as .keras file.

Run: python train_cnn.py (This is your "Part 2" script)

Output: waste_cnn_model.keras

2. Train the SVM Model (Stage 2)
This script creates the simple text-based classifier. It defines the mapping from a specific item to a broad category (e.g., "apple" → "Organic Waste") and trains an SVM on this map.

Run: python train_svm.py (This is your "Part 3" script)

Output: svm_text_pipeline.joblib

3. Run Predictions
This is the final application. It loads both trained models (.keras and .joblib) and provides a simple interface to classify new images.

Run: python predict.py (This is your final script)

Action: The script will ask you to upload one or more images.

Result: It will print the classification for each image to the console.

📈 Improving Performance
If you find the model is making frequent mistakes (misclassifications), the problem is almost always in the Stage 1 (CNN) Model. The original model was trained with a small dataset and few epochs.

To improve accuracy, re-train the CNN using these techniques:

Train for More Epochs: In your CNN training script, increase epochs=10 to a higher number, like 30 or 50.

Use Data Augmentation: Add more augmentation layers (RandomZoom, RandomContrast, etc.) to your Keras Sequential model to "fake" more training data and prevent overfitting.

Implement Fine-Tuning: After an initial training phase, "unfreeze" the top layers of the MobileNetV2 base model and continue training with a very low learning rate (1e-5). This allows the model to learn more specific features from your waste images.
