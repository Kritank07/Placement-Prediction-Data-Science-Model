# Placement-Prediction-Data-Science-Model
This project predicts student placement outcomes based on academic and profile data using a trained Random Forest Classifier. It uses a complete machine learning pipeline with preprocessing, label encoding, and model persistence using `joblib`.

## Files and Structure
1. placement.csv # Original dataset
2. input.csv # Automatically generated test data (from StratifiedShuffleSplit)
3. output.csv # Output file with predictions
4. model.pkl # Trained Random Forest model
5. preprocessing_pipeline.pkl # Preprocessing pipeline (numerical + categorical)
6. label_encoder.pkl # Encoder to convert labels to/from numeric

## How the Model Works

1. Loads `placement.csv` as the dataset. (CSV to be downloaded from kaggle)
2. Performs a **StratifiedShuffleSplit** to ensure class distribution is preserved.
3. Splits the data into training and test sets.
4. Stores the test set as `input.csv` (used later for inference).
5. Encodes the target labels using `LabelEncoder`.
6. Separates numerical and categorical columns.
7. Constructs a preprocessing pipeline using:
   - `SimpleImputer` and `StandardScaler` for numerical data.
   - `OneHotEncoder` for categorical data.
8. Trains a `RandomForestClassifier` on the transformed training data.
9. Saves the trained model, preprocessing pipeline, and label encoder using `joblib`.

On rerunning the script:
- If model files exist, the script loads the model and pipeline.
- Reads `input.csv`, applies the pipeline, and predicts outcomes.
- Saves the result to `output.csv` with an additional column `Placement_Prediction`.

## Requirements
- pandas
- numpy
- scikit-learn
- joblib

## Highlights
1. Stratified sampling ensures balanced training.
2. Full pipeline for preprocessing both numerical and categorical features.
3. Label encoding ensures clean conversion between text and numbers.
4. Model persistence using joblib ensures easy inference without retraining.
