# ToxiTracAI 

ToxiTracAI is a school project aimed at developing a machine-learning system for detecting alcohol consumption based on heart rate data. The project encompasses data preparation, model training, and application through a neural network.

## Table of Contents 

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notes on Data and Model Organization](#notes-on-data-and-model-organization)
- [Future Developments](#future-developments)

---

## Overview 

In this project, we analyze heart rate data from intoxicated and non-intoxicated individuals to create a machine-learning model for detecting changes in heart rate.  
### Features: 
- Data Preparation: Consolidation and processing of raw data. 
- Model Training: Neural network for classifying normal and abnormal states. 
- Easy application via `main.py`.

---

## Project Structure 

Here is an overview of the project structure: 

```
ToxiTracAI/
├── datasets/                # Raw and processed datasets 
├── model-training/          # Notebooks for model training 
├── models/                  # Model and scaler files 
│   ├── heart_nn_normalrate_model2.keras  # Trained model 
│   └── scaler_nn_model2.joblib            # Scaler for data normalization 
├── wavelet-transform/       # Experiments for data analysis 
├── .gitignore               # Ignored files and folders 
├── config.py                # Central configuration file for paths 
├── main.py                  # Main script for application 
├── toxitracai.py            # Main class for analysis 
├── README.md                # Project description 
└── requirements.txt         # Dependencies for the project 
```

---

## Installation 

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/prenyx/ToxiTracAI.git
   cd ToxiTracAI
   ```

2. **Create and Activate a Virtual Environment:**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate      # For Windows
   ```

3. **Install Required Packages:**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage 

1. **Data Preparation:**  
   In the notebook `data_preparation.ipynb`, the heart rate data from intoxicated and non-intoxicated individuals has been consolidated. However, this data is not yet fully prepared; only the **normal BPM data** is used.

2. **Model Application:**  
   To run the program, simply execute the `main.py` file. This initializes the **ToxitracAI class**, which loads the prepared model and makes predictions based on the input BPM data.  
   ```bash
   python main.py
   ```

3. **Example Output:**  
   - Input the minimum and maximum BPM values as well as the current condition (e.g., "resting").  
   - The model will return whether the state is recognized as "active" or "normal."

---

## Notes on Data and Model Organization 

1. **Model Files:**  
   Model files are located in the `models/` folder, including the trained model (`heart_nn_normalrate_model2.keras`) and the scaler (`scaler_nn_model2.joblib`).  
   All model paths are dynamically configured through the `config.py` file, allowing them to be used independently of the system.

2. **Dataset Folder:**  
   The `datasets/` folder contains some example datasets that can be used for evaluating the model.

3. **Wavelet Data:**  
   The `wavelet-transform/` folder contains experimental EKG data for **normal BPM values**, which could be expanded in future iterations of the project.

4. **Model Training:**  
   The `model-training/` folder contains all relevant codes and notebooks for training the neural network, including data splitting, feature scaling, and model optimization.

---

## Future Developments 

- Integration of intoxicated data into model training. 
- Validation of the model using a larger and more diverse dataset. 
- Optimization of the neural network to improve accuracy.

---

# Best of luck using ToxiTracAI! 🎉 

--- 
