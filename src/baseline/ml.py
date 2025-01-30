import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def encode_categorical_columns(X):
    """Encode categorical columns in the dataset using Label Encoding."""
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    return X, label_encoders


def preprocess_data(X, y):
    """Preprocess features and target."""
    # Encode categorical columns in features
    X, label_encoders = encode_categorical_columns(X)

    # Fill missing values with the column mean
    X = X.fillna(X.mean())

    # Encode target column if it is categorical
    y = LabelEncoder().fit_transform(y.astype(str))
    return X, y


def create_pipelines():
    """Create machine learning pipelines with preprocessing."""
    base_pipeline = [('scaler', MinMaxScaler())]

    models = {
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gaussian NB': GaussianNB(var_smoothing=1e-02),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    }

    return {name: Pipeline(base_pipeline + [('classifier', model)]) for name, model in models.items()}


def evaluate_models(X, y, pipelines, cv=2):
    """Evaluate models using cross-validation."""
    results_list = []
    scoring = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']

    os.makedirs('results/baseline', exist_ok=True)

    for model_idx, (model_name, pipeline) in enumerate(pipelines.items(), 1):
        print(f"\n[{model_idx}/{len(pipelines)}] Training {model_name}")
        try:
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
            result = {
                'accuracy': scores['test_accuracy'].mean(),
                'f1_score': scores['test_f1_weighted'].mean(),
                'recall': scores['test_recall_weighted'].mean(),
                'precision': scores['test_precision_weighted'].mean(),
                'modelo_ML': model_name,
            }
            results_list.append(result)

            # Save individual model results
            model_file = f'results/baseline/{model_name.replace(" ", "_")}_results.csv'
            df = pd.DataFrame([result])
            if not os.path.exists(model_file):
                df.to_csv(model_file, index=False)
            else:
                df.to_csv(model_file, mode='a', header=False, index=False)

            print(f"Completed {model_name} - Accuracy: {scores['test_accuracy'].mean():.3f}")
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")

    return pd.DataFrame(results_list)


def main():
    print("Starting pipeline execution...")
    file1 = 'data/Attack.csv'
    file2 = 'data/environmentMonitoring.csv'
    file3 = 'data/patientMonitoring.csv'

    # Load datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Ensure all datasets have the same columns
    all_columns = set(df1.columns) | set(df2.columns) | set(df3.columns)
    df1 = df1.reindex(columns=all_columns, fill_value=None)
    df2 = df2.reindex(columns=all_columns, fill_value=None)
    df3 = df3.reindex(columns=all_columns, fill_value=None)

    # Concatenate datasets for the baseline
    dataset = pd.concat([df1, df2, df3], ignore_index=True)

    # Separate features and target
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Preprocess data
    X, y = preprocess_data(X, y)

    # Create pipelines and evaluate models
    pipelines = create_pipelines()
    results_df = evaluate_models(X, y, pipelines)

    # Save consolidated results
    results_df.to_csv('results/baseline/all_results.csv', index=False)
    print("\nResults saved in results/baseline/")


if __name__ == "__main__":
    main()
