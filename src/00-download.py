import os
import gdown
import config
from utils import get_logger

logger = get_logger("DATA DOWNLOADING")

def download_data():
    output_folder = config.DATA_DIR  
    
    #https://drive.google.com/file/d/1e1WL4cdU62ifylNOd8zVawCXRZkCavZ4/view?usp=sharing
    #https://drive.google.com/file/d/1uBkJlXkPErRKf8t6A9rmQU546HZlBtqn/view?usp=sharing
    data_id = '1uBkJlXkPErRKf8t6A9rmQU546HZlBtqn' 
    stops_id = '1e1WL4cdU62ifylNOd8zVawCXRZkCavZ4'
    
    data_path = os.path.join(output_folder, 'raw_data.csv')
    
    stops_path = os.path.join(output_folder, 'stop_times.txt')
    
    if os.path.exists(data_path) and os.path.exists(stops_path):
        logger.info(f"Data already exists in '{output_folder}' folder. ")
        logger.info("Skipping download...")
        return

    # Mappa létrehozása
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logger.info(f"Downloading data from Google Drive...")
    
    url = f'https://drive.google.com/uc?id={data_id}'
    gdown.download(url, data_path, quiet=False)

    url = f'https://drive.google.com/uc?id={stops_id}'
    gdown.download(url, stops_path, quiet=False)
    
    logger.info("Download success!")

if __name__ == "__main__":
    download_data()