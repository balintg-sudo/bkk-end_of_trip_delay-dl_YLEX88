# src/config.py
import os

DATA_DIR = 'data'
MODEL_DIR = 'models'

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')

DAYS_TO_TRAIN = 11

LOW_MEMORY = True
# Hyperparameters
EPOCHS = 5         
BATCH_SIZE = 2048   
LEARNING_RATE = 0.005