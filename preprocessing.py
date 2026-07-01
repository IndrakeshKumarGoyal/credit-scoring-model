import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CreditPreprocessor:
    def __init__(self, dataset_type='synthetic'):
        self.dataset_type = dataset_type
        self.preprocessor = None
        self.feature_names = None
        self.engineered_feature_names = []

    def engineer_features(self, df):
        """
        Engineers features based on the dataset type.
        """
        df_engineered = df.copy()
        
        if self.dataset_type == 'synthetic':
            # 1. Debt to Income Ratio
            df_engineered['Debt_to_Income_Ratio'] = df_engineered['Total_Debt'] / (df_engineered['Annual_Income'] + 1e-5)
            # 2. Payment to Income Ratio (Annualized payment / annual income)
            df_engineered['Payment_to_Income_Ratio'] = (df_engineered['Monthly_Payment'] * 12) / (df_engineered['Annual_Income'] + 1e-5)
            # 3. Savings to Income Ratio
            df_engineered['Savings_to_Income_Ratio'] = df_engineered['Savings_Balance'] / (df_engineered['Annual_Income'] + 1e-5)
            
            self.engineered_feature_names = ['Debt_to_Income_Ratio', 'Payment_to_Income_Ratio', 'Savings_to_Income_Ratio']
            
        elif self.dataset_type == 'german_credit':
            # 1. Credit to Age Ratio
            df_engineered['Credit_to_Age_Ratio'] = df_engineered['Credit_Amount'] / (df_engineered['Age'] + 1e-5)
            # 2. Installment to Income proxy: Installment_Rate is 1,2,3,4.
            # In German credit, installment_commitment is percentage of disposable income
            # We can also add savings to credit ratio
            # Let's create Savings_to_Credit_Ratio if we can map savings status
            self.engineered_feature_names = ['Credit_to_Age_Ratio']
            
        return df_engineered

    def fit_transform(self, df):
        """
        Engineers features, identifies column types, fits the ColumnTransformer, and returns processed X and y.
        """
        # 1. Engineer features
        df_engineered = self.engineer_features(df)
        
        # 2. Separate features and target
        if 'Credit_Default' in df_engineered.columns:
            X = df_engineered.drop(columns=['Credit_Default'])
            y = df_engineered['Credit_Default']
        else:
            X = df_engineered.copy()
            y = None
            
        # 3. Identify column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        
        logging.info(f"Preprocessing {self.dataset_type} - Categorical: {categorical_cols}, Numerical: {numerical_cols}")
        
        # 4. Set up transformers
        # For numerical features: impute missing values (if any) with median
        numerical_transformer = SimpleImputer(strategy='median')
        
        # For categorical features: impute missing with most frequent and one-hot encode
        categorical_transformer = ColumnTransformer(
            transformers=[
                ('imputer', SimpleImputer(strategy='most_frequent'), categorical_cols),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )
        # Wait, using ColumnTransformer inside ColumnTransformer is possible, but we can do standard pipeline:
        from sklearn.pipeline import Pipeline
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ]
        )
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get output feature names for visualization
        self.feature_names = self._get_feature_names(numerical_cols, categorical_cols)
        
        # Convert back to DataFrame
        X_df = pd.DataFrame(X_processed, columns=self.feature_names, index=df.index)
        
        return X_df, y

    def transform(self, df):
        """
        Applies engineered features and the fitted ColumnTransformer to new data.
        """
        if self.preprocessor is None:
            raise ValueError("The preprocessor must be fitted using fit_transform before calling transform.")
            
        df_engineered = self.engineer_features(df)
        
        if 'Credit_Default' in df_engineered.columns:
            X = df_engineered.drop(columns=['Credit_Default'])
        else:
            X = df_engineered.copy()
            
        X_processed = self.preprocessor.transform(X)
        
        X_df = pd.DataFrame(X_processed, columns=self.feature_names, index=df.index)
        return X_df

    def _get_feature_names(self, numerical_cols, categorical_cols):
        """
        Helper to extract names from fitted preprocessor.
        """
        # Numerical columns retain their names
        names = list(numerical_cols)
        
        # For categorical columns, get the one-hot encoded names
        if categorical_cols:
            cat_pipeline = self.preprocessor.named_transformers_['cat']
            onehot = cat_pipeline.named_steps['onehot']
            # Get category names
            cat_features = onehot.get_feature_names_out(categorical_cols)
            names.extend(cat_features)
            
        return names

def prepare_train_test_data(df, dataset_type='synthetic', test_size=0.2, random_state=42):
    """
    Convenience function to preprocess a dataset and split into train/test sets.
    """
    preprocessor = CreditPreprocessor(dataset_type)
    X, y = preprocessor.fit_transform(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    from src.data_loader import generate_synthetic_credit
    df = generate_synthetic_credit(10)
    X_train, X_test, y_train, y_test, prep = prepare_train_test_data(df)
    print("Preprocessed Columns:\n", X_train.columns.tolist()[:10])
    print("Preprocessed Shape:", X_train.shape)
