import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load Data
def load_data(raw_sales_path, ma_lga_path):
    # Load CSV files into pandas DataFrames
    raw_sales = pd.read_csv(raw_sales_path)
    ma_lga = pd.read_csv(ma_lga_path)
    return raw_sales, ma_lga

# Preprocessing raw_sales data
def preprocess_raw_sales(raw_sales):
    # Drop unnecessary rows or columns with summaries, if any
    raw_sales = raw_sales.dropna(subset=['datesold', 'price', 'postcode', 'propertyType', 'bedrooms'])

    # Convert datesold to datetime
    raw_sales['datesold'] = pd.to_datetime(raw_sales['datesold'], errors='coerce')

    # Convert price to numeric, removing commas and handling errors
    raw_sales['price'] = raw_sales['price'].replace('[\$,]', '', regex=True).astype(float)

    # Encode categorical columns (propertyType)
    le = LabelEncoder()
    raw_sales['propertyType'] = le.fit_transform(raw_sales['propertyType'])

    # Handle missing values for numerical columns (e.g., median imputation or drop)
    raw_sales = raw_sales.dropna()

    # Aggregate or transform as needed for time-series analysis (e.g., monthly mean price)
    raw_sales['month'] = raw_sales['datesold'].dt.to_period('M')
    time_series = raw_sales.groupby('month')['price'].mean().reset_index()
    time_series['month'] = time_series['month'].dt.to_timestamp()

    return time_series

# Preprocessing ma_lga data
def preprocess_ma_lga(ma_lga):
    # Clean saledate column and convert to datetime
    ma_lga = ma_lga.dropna(subset=['saledate', 'MA'])
    ma_lga['saledate'] = pd.to_datetime(ma_lga['saledate'], errors='coerce')

    # Convert MA to numeric, removing percentages if needed
    ma_lga['MA'] = ma_lga['MA'].replace('%', '', regex=True).astype(float)

    # Encode property type
    le = LabelEncoder()
    ma_lga['type'] = le.fit_transform(ma_lga['type'])

    # Aggregate data by date if needed
    ma_lga = ma_lga.groupby('saledate').agg({'MA': 'mean', 'type': 'first', 'bedrooms': 'first'}).reset_index()

    return ma_lga

# Combine datasets for HMM
def transform_for_hmm(time_series, ma_lga):
    # Merge datasets on the closest date or using a time-based join
    combined = pd.merge_asof(
        time_series.sort_values('month'),
        ma_lga.sort_values('saledate'),
        left_on='month',
        right_on='saledate',
        direction='nearest'
    )

    # Drop unnecessary columns for modeling
    combined = combined.drop(columns=['month', 'saledate'])

    # Scale features for HMM
    scaler = MinMaxScaler()
    combined_scaled = pd.DataFrame(scaler.fit_transform(combined), columns=combined.columns)

    return combined_scaled

# Exploratory Data Analysis (EDA)
def perform_eda(raw_sales, ma_lga):
    print("\n--- Raw Sales Data Info ---")
    print(raw_sales.info())
    print("\n--- MA LGA Data Info ---")
    print(ma_lga.info())

    # Plot time-series data for exploration
    raw_sales['datesold'] = pd.to_datetime(raw_sales['datesold'], errors='coerce')
    raw_sales = raw_sales.dropna(subset=['datesold', 'price'])
    plt.figure(figsize=(10, 6))
    raw_sales.groupby(raw_sales['datesold'].dt.to_period('M'))['price'].mean().plot(title='Average Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.show()

# Main function to execute the workflow
def main():
    raw_sales_path = 'raw_sales.csv'
    ma_lga_path = 'ma_lga_12345.csv'

    # Load datasets
    raw_sales, ma_lga = load_data(raw_sales_path, ma_lga_path)

    # Preprocess datasets
    time_series = preprocess_raw_sales(raw_sales)
    ma_lga_cleaned = preprocess_ma_lga(ma_lga)

    # Perform EDA
    perform_eda(raw_sales, ma_lga)

    # Prepare data for HMM
    hmm_data = transform_for_hmm(time_series, ma_lga_cleaned)
    print("\n--- Transformed Data for HMM ---")
    print(hmm_data.head())

    # Save preprocessed data for future use
    hmm_data.to_csv('hmm_ready_data.csv', index=False)

if __name__ == '__main__':
    main()
