from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import joblib
import os

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "preprocessing_pipeline.pkl"
ENCODER_FILE = "label_encoder.pkl"  # optional, if you want to decode prediction

if not os.path.exists(MODEL_FILE):
    # Train the model
    df = pd.read_csv("placement.csv")   

    #strattified shuffle split
    splits = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
    for train_index, test_index in splits.split(df, df['Placement']):
        strat_train_set = df.loc[train_index]
        df.loc[test_index].to_csv("input.csv", index=False)
    
    # make copy of the Training set
    train_set_copy = strat_train_set.copy()

    # Seperate Features and labels
    labels = train_set_copy["Placement"]
    features = train_set_copy.drop("Placement", axis=1)

    #convert labels also to numerical values
    label_encoder = LabelEncoder()  # Added: create label encoder instance
    labels = label_encoder.fit_transform(labels)

    # Print labels and features

    # Seperate numerical and catagorical attributes for train set
    num_attribs_train = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    column_attribs_train = features.select_dtypes(include=["object"]).columns.tolist()

    # Create a pipeline for numerical attributes
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy = "median")),
            ("standardization", StandardScaler())
        ]
    )
    cat_pipeline = Pipeline([
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    #combine numerical and categorical pipelines
    full_pipeline = ColumnTransformer([
        ('nums', num_pipeline, num_attribs_train),
        ('cats', cat_pipeline, column_attribs_train)
    ])

    # num and cat are just the names of the transformers, you can name them anything you want.

    final_placement = full_pipeline.fit_transform(features)

    # Train a Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(final_placement, labels)
    predictions = model.predict(final_placement)
    random_rmses = cross_val_score(model, final_placement, labels, scoring="accuracy", cv=10)
    # print(pd.Series(random_rmses).describe())

    # Save the model and preprocessing pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(full_pipeline, PIPELINE_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)


else:

    # Load model and preprocessing pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    label_encoder = joblib.load(ENCODER_FILE)

    # Load input data
    input_data = pd.read_csv("input.csv")

    # Apply transformations
    transformed_input = pipeline.transform(input_data)

    # Predict placement
    predictions = model.predict(transformed_input)

    # Decode labels (optional)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Add predictions to the input data
    input_data["Placement_Prediction"] = predicted_labels

    # Save output
    try:
        input_data.to_csv('output.csv', index=False)
        print('File saved successfully')
    except Exception as e:
        print(f"Failed to save file: {e}")
