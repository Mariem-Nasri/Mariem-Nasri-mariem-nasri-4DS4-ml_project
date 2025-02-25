import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

# Paths for saving data
from src.config import DATA_PATHS  # Assuming paths are imported from main.py

def prepare_data():
    df = pd.read_csv('~/mariem-nasri-4DS4-ml_project/data/data_churn.csv')
    df_dp = df.copy()

    # Handling outliers
    columns_outliers = ['Total eve calls', 'Total day calls', 'Total intl calls']
    for column in columns_outliers:
        Q1, Q3 = df_dp[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_dp[column] = df_dp[column].clip(lower=lower_bound, upper=upper_bound)

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    df_dp['International plan'] = label_encoder.fit_transform(df_dp['International plan'])
    df_dp['Voice mail plan'] = label_encoder.fit_transform(df_dp['Voice mail plan'])
    df_dp['Churn'] = label_encoder.fit_transform(df_dp['Churn'])

    # Feature Engineering: State Churn Rate
    state_churn_rate = df_dp.groupby('State')['Churn'].mean().reset_index()
    state_churn_rate.rename(columns={'Churn': 'Churn_Rate'}, inplace=True)
    kmeans = KMeans(n_clusters=3, random_state=42)
    state_churn_rate['Cluster'] = kmeans.fit_predict(state_churn_rate[['Churn_Rate']].values)

    cluster_mapping = (
        state_churn_rate.groupby('Cluster')['Churn_Rate']
        .mean()
        .sort_values()
        .index.to_list()
    )
    cluster_labels = {cluster_mapping[0]: 'Low', cluster_mapping[1]: 'Medium', cluster_mapping[2]: 'High'}
    state_churn_rate['State_Category'] = state_churn_rate['Cluster'].map(cluster_labels)
    df_dp = df_dp.merge(state_churn_rate[['State', 'State_Category']], on='State', how='left')
    df_dp['State_Category'] = df_dp['State_Category'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df_dp.drop(columns=['State'], inplace=True)

    # Dropping insignificant columns
    df_dp.drop(columns=['Area code', 'Voice mail plan'], inplace=True)
    df_dp.drop(columns=['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes'], inplace=True)

    # Feature Engineering: Usage Score
    corr_features = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    weights = np.abs(df_dp[corr_features].corrwith(df_dp['Churn']))
    weights /= weights.sum()
    df_dp['Usage Score'] = df_dp[corr_features].mul(weights).sum(axis=1)

    # Splitting data
    X = df_dp.drop(columns=['Churn'])
    y = df_dp['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data scaling
    scaler = StandardScaler()
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Address Class Imbalance (only for training data)
    smote = SMOTE(random_state=42)
    X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Save data to disk
    joblib.dump(X_train_scaled_smote, DATA_PATHS['X_train'])
    joblib.dump(X_test_scaled, DATA_PATHS['X_test'])
    joblib.dump(y_train_smote, DATA_PATHS['y_train'])
    joblib.dump(y_test, DATA_PATHS['y_test'])

    return X_train_scaled_smote, X_test_scaled, y_train_smote, y_test
