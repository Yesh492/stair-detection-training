# Mobile Stair and Step Detection for Blind Navigation  
### CS663 – Project 2 | Group 8  
Author: Yeshvanth Kumar Domala  

## Objective  
The goal of this project is to develop a mobile-based vision system to detect stairs and steps in real-time to aid blind and low-vision individuals. The system uses computer vision techniques to analyze RGB images from a mobile camera and will eventually provide feedback through audio or vibration.  

## Dataset  
- **Source:** RGB-D Stair Dataset (Mendeley Data).  
- Contains RGB and depth images with labeled stair and flat scenes. Only the RGB portion is used here.  

## Week 1–2 Progress  
- Mounted dataset from Google Drive in Colab and explored the structure (train, val, test).  
- Created a subset of the dataset for quick experiments.  
- Implemented classical edge-based stair detection using Canny edge detector and Hough line transform.  
- Started building a lightweight CNN architecture to classify stair images.  

## Repository Structure  
- `notebooks/` – Colab notebooks for training and experiments.  
- `outputs/` – Contains outputs from classical approaches and other results.  
- `models/` – Trained model files (.h5 and .tflite).  
- `wiki/` – Project wiki pages and progress reports.  

## Next Steps  
- Separate data into stair_up, stair_down, and flat classes.  
- Train a CNN classifier and convert it to TensorFlow Lite for mobile deployment.  
- Create an Android app that performs real-time stair detection and provides user feedback.
