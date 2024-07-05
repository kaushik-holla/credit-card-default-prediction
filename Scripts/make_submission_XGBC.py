import pandas as pd
import joblib

def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """
    return joblib.load(model_path)

def preprocess_data(data_path, preprocessor):
    """
    Preprocess the data using the provided preprocessor pipeline.
    """
    # Load the test data
    data = pd.read_csv(data_path)
    
    # Ensure that the 'customer_id' column is not included in the features
    customer_ids = data['customer_id']
    X = data.drop('customer_id', axis=1)  # Drop 'customer_id' if it's present
    
    # Apply preprocessing
    X_preprocessed = preprocessor.transform(X)
    
    return customer_ids, X_preprocessed

def make_predictions(model, data):
    """
    Make predictions using the provided model and preprocessed data.
    """
    return model.predict_proba(data)[:, 1]

def save_predictions(customer_ids, predictions, output_path):
    """
    Save the predictions to a CSV file in the required format.
    """
    submission = pd.DataFrame({
        'customer_id': customer_ids,
        'pr_y': predictions
    })
    submission.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Define paths
    model_path = 'Model/XGBoostClassifier_pipeline.joblib'  # Path to the saved model
    test_data_path = 'Data/test.csv'  # Path to the test data
    output_path = 'Prediction/submission.csv'  # Output path for the predictions CSV
    
    # Load the pre-trained model
    pipeline = load_model(model_path)
    
    # Preprocess the test data
    customer_ids, X_test_preprocessed = preprocess_data(test_data_path, pipeline.named_steps['preprocessor'])
    
    # Make predictions
    predictions = make_predictions(pipeline.named_steps['classifier'], X_test_preprocessed)
    
    # Save the predictions
    save_predictions(customer_ids, predictions, output_path)

    print(f"Predictions saved to {output_path}")
