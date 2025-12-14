import pandas as pd
import numpy as np
import os
import sys
from utils import get_logger
import config

logger = get_logger("DATA PREPROCESSING")

# Config importálása a helyes elérési utakhoz
try:
    import config
    RAW_DATA_PATH = os.path.join(config.DATA_DIR, 'raw_data.csv')
    TRAIN_DATA_PATH = config.TRAIN_DATA_PATH
    TEST_DATA_PATH = config.TEST_DATA_PATH
except ImportError:
    RAW_DATA_PATH = 'data/raw_data.csv'
    TRAIN_DATA_PATH = 'data/train_data.csv'
    TEST_DATA_PATH = 'data/test_data.csv'

def clean_data(df):
    logger.info(f"Original size: {len(df):,} rows")

    df = df.drop_duplicates(subset=['trip_id', 'timestamp_utc'])

    df = df[(df['delay_seconds'] > -900) & (df['delay_seconds'] < 7200)]

    if 'lat' in df.columns and 'lon' in df.columns:
        df = df[(df['lat'] > 47.1) & (df['lat'] < 47.7) & 
                (df['lon'] > 18.8) & (df['lon'] < 19.3)]
        
    trip_counts = df['trip_id'].value_counts()
    valid_trips = trip_counts[trip_counts >= 10].index
    df = df[df['trip_id'].isin(valid_trips)]
    
    logger.info(f"Cleaned size: {len(df):,} rows")
    return df

def main():
    logger.info("---         STARTING PREPROCESSING          ---")
    logger.info("================ CONFIGURATION ================")
    logger.info(f"Days to Train:   {config.DAYS_TO_TRAIN}")
    logger.info(f"Low Memory:      {config.LOW_MEMORY}")
    logger.info(f"Train Data Path: {TRAIN_DATA_PATH}")
    logger.info(f"Test Data Path:  {TEST_DATA_PATH}")
    logger.info("===============================================")
    
    logger.info(f"Reading data from {RAW_DATA_PATH}...")
    if config.LOW_MEMORY:
        data_df = pd.read_csv(RAW_DATA_PATH, dtype={
            'trip_id': 'str',
            'route_id': 'str',
            'stop_id': 'str',
            'delay_seconds': 'float32',
            'lat': 'float32',
            'lon': 'float32',
            'speed': 'float32',
            'heading': 'float32'
        }, nrows=5000000, low_memory=True)
    else:
        data_df = pd.read_csv(RAW_DATA_PATH, dtype={
            'trip_id': 'str',
            'route_id': 'str',
            'stop_id': 'str',
            'delay_seconds': 'float32',
            'lat': 'float32',
            'lon': 'float32',
            'speed': 'float32',
            'heading': 'float32'
        })

    logger.info("Feature Engineering...")
    data_df['timestamp_utc'] = pd.to_datetime(data_df['timestamp_utc'], errors='coerce')
    
    data_df['hour'] = data_df['timestamp_utc'].dt.hour
    data_df['day_of_week'] = data_df['timestamp_utc'].dt.dayofweek 
    data_df['is_weekend'] = data_df['day_of_week'] >= 5
    
    # Sin/Cos
    data_df['hour_sin'] = np.sin(2 * np.pi * data_df['hour'] / 24)
    data_df['hour_cos'] = np.cos(2 * np.pi * data_df['hour'] / 24)
    data_df['day_sin'] = np.sin(2 * np.pi * data_df['day_of_week'] / 7)
    data_df['day_cos'] = np.cos(2 * np.pi * data_df['day_of_week'] / 7)
    
    logger.info("Calculating final_delay...")
    data_df = data_df.sort_values(['trip_id', 'timestamp_utc'])
    data_df['final_delay'] = data_df.groupby('trip_id')['delay_seconds'].transform('last')

    logger.info("Cleaning data...")
    df_cleaned = clean_data(data_df)
    
    del data_df 
    
    FEATURES_TO_KEEP = [
        'trip_id', 
        'timestamp_utc', 
        'final_delay',
        'delay_seconds',
        'current_stop_sequence',
        'speed',
        'heading',
        'lat',
        'lon',
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'is_weekend',
        'route_id',
        'stop_id'
    ]
    
    available_cols = [c for c in FEATURES_TO_KEEP if c in df_cleaned.columns]
    df_final = df_cleaned[available_cols].copy()
    
    del df_cleaned
    
    df_final['is_weekend'] = df_final['is_weekend'].astype(int)

    logger.info("Splitting Train/Test...")
    
    df_final = df_final.sort_values('timestamp_utc')
    df_final['process_date'] = df_final['timestamp_utc'].dt.date
    
    unique_dates = sorted(df_final['process_date'].unique())
    
    TRAIN_DAYS_COUNT = config.DAYS_TO_TRAIN
    split_index = TRAIN_DAYS_COUNT
    
    if len(unique_dates) <= TRAIN_DAYS_COUNT:
        logger.info("Warning: Less than 10 days of data. Using 80% split.")
        split_index = int(len(unique_dates) * 0.8)

    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]
    
    logger.info(f"Train dates count: {len(train_dates)}")
    logger.info(f"Test dates count:  {len(test_dates)}")

    logger.info("Saving files...")
    
    train_mask = df_final['process_date'].isin(train_dates)
    train_df = df_final[train_mask].drop(columns=['process_date'])
    
    logger.info(f"Saving TRAIN ({len(train_df):,} rows) to {TRAIN_DATA_PATH}")
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    del train_df 
    
    test_mask = df_final['process_date'].isin(test_dates)
    test_df = df_final[test_mask].drop(columns=['process_date'])
 
    logger.info(f"Saving TEST ({len(test_df):,} rows) to {TEST_DATA_PATH}")
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    logger.info("Preprocessing DONE.")

if __name__ == "__main__":
    main()