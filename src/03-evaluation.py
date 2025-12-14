import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, max_error
from torch.utils.data import TensorDataset, DataLoader
import os
import joblib
import config
from utils import get_logger, BKK_GNN_Weighted
import torch

logger = get_logger("EVALUATION")

def load_saved_graph_artifacts(device):
    logger.info("Loading pre-built graph artifacts...")
    try:
        edge_index = torch.load(
            os.path.join(config.MODEL_DIR, 'edge_index.pt'), 
            map_location=device, 
            weights_only=True
        )
        edge_weight = torch.load(
            os.path.join(config.MODEL_DIR, 'edge_weight.pt'), 
            map_location=device, 
            weights_only=True  
        )
        x = torch.load(
            os.path.join(config.MODEL_DIR, 'node_features.pt'), 
            map_location=device, 
            weights_only=True 
        )
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        x = x.to(device)
        
        return x, edge_index, edge_weight
        
    except FileNotFoundError as e:
        logger.error(f"Missing graph artifact: {e}")
        logger.error("Please run 02-training.py first to generate these files!")
        raise e

def calculate_metrics(name, targets, preds):
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    medae = median_absolute_error(targets, preds)
    max_err = max_error(targets, preds)

    errors_sec = np.abs(targets - preds)

    acc_1min = np.mean(errors_sec <= 60) * 100
    acc_3min = np.mean(errors_sec <= 180) * 100
    acc_5min = np.mean(errors_sec <= 300) * 100
    
    logger.info(f"--- {name} RESULTS ---")
    logger.info(f"MAE:                     {mae/60:.2f} min")
    logger.info(f"RMSE:                    {rmse/60:.2f} min")
    logger.info(f"MedAE:                   {medae/60:.2f} min")
    logger.info(f"Max Error:               {max_err/60:.1f} min")
    logger.info("-" * 30)
    logger.info(f"Accuracy +/- 1 minutes:  {acc_1min:.1f}%")
    logger.info(f"Accuracy +/- 3 minutes:  {acc_3min:.1f}%")
    logger.info(f"Accuracy +/- 5 minutes:  {acc_5min:.1f}%")
    logger.info("=" * 40)
    return mae, acc_3min

def main():
    logger.info("--- STARTING EVALUATION ON TEST  ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        le = joblib.load(os.path.join(config.MODEL_DIR, 'label_encoder.pkl'))
        x, edge_index, edge_weight = load_saved_graph_artifacts(device)
    except FileNotFoundError:
        logger.error("Files missing. Run training first!")
        return

    gnn_model = BKK_GNN_Weighted(num_node_features=2, hidden_channels=64).to(device)
    gnn_path = os.path.join(config.MODEL_DIR, 'best_gnn_model.pth')
    if os.path.exists(gnn_path):
        gnn_model.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
        gnn_model.eval()
    else:
        logger.error("GNN model missing!")
        return

    lr_model = None
    lr_path = os.path.join(config.MODEL_DIR, 'linear_regression.joblib')
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)

    logger.info(f"Loading Test Data: {config.TEST_DATA_PATH}")
    df_test = pd.read_csv(config.TEST_DATA_PATH, dtype={'stop_id': str})
    df_test = df_test.dropna(subset=['final_delay'])
    
    known_stops = set(le.classes_)
    df_test = df_test[df_test['stop_id'].isin(known_stops)]
    
    df_test['stop_idx'] = le.transform(df_test['stop_id'])
    
    feature_cols = ['delay_seconds', 'speed', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'heading']
    df_test[feature_cols] = df_test[feature_cols].fillna(0)
    
    X_test = torch.tensor(df_test[feature_cols].values, dtype=torch.float)
    y_test = torch.tensor(df_test['final_delay'].values, dtype=torch.float).view(-1, 1)
    stop_idx_test = torch.tensor(df_test['stop_idx'].values, dtype=torch.long)
    
    test_loader = DataLoader(TensorDataset(X_test, y_test, stop_idx_test), batch_size=config.BATCH_SIZE, shuffle=False)
    
    gnn_preds = []
    lr_preds = []
    heuristic_preds = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_idx in test_loader:
            b_X_gpu = batch_X.to(device)
            b_idx_gpu = batch_idx.to(device)
            
            # GNN
            gnn_preds.append(gnn_model(x, edge_index, edge_weight, b_X_gpu, b_idx_gpu).cpu())
            
            # LR
            if lr_model:
                lr_preds.append(torch.tensor(lr_model.predict(batch_X.numpy())))
            
            # Heuristic
            heuristic_preds.append(batch_X[:, 0].view(-1, 1))
            targets.append(batch_y)
            
    # Concat
    gnn_preds = torch.cat(gnn_preds).numpy()
    heuristic_preds = torch.cat(heuristic_preds).numpy()
    targets = torch.cat(targets).numpy()
    if lr_model:
        lr_preds = torch.cat(lr_preds).numpy()

    # 5. EREDMÉNYEK
    print("\n" + "="*50)
    print("         DETAILED MODEL COMPARISON")
    print("="*50)
    
    calculate_metrics("1. NAIVE BASELINE", targets, heuristic_preds)
    
    if lr_model:
        calculate_metrics("2. LINEAR REGRESSION", targets, lr_preds)
        
    mae_gnn, acc3_gnn = calculate_metrics("3. GNN MODEL", targets, gnn_preds)

if __name__ == "__main__":
    main()