import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib
import time
import gc
import config
from utils import get_logger, BKK_GNN_Weighted

logger = get_logger("TRAINING")

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params

def parse_gtfs_time(time_str):
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return np.nan

def build_static_graph_and_cleanup():
    logger.info("---          Building Static Graph from stop_times.txt ---")
    
    stop_times_path = 'data/stop_times.txt'
    if not os.path.exists(stop_times_path):
        stop_times_path = os.path.join(os.getcwd(), 'data', 'stop_times.txt')

    logger.info("Loading stop_times.txt...")

    st_df = pd.read_csv(stop_times_path, 
                        usecols=['trip_id', 'stop_id', 'stop_sequence', 'arrival_time'],
                        dtype={'stop_id': str})
    
    logger.info(f"Stop times loaded ({len(st_df):,} rows). Processing edges...")
    
    st_df['arrival_sec'] = st_df['arrival_time'].apply(parse_gtfs_time)
    st_df = st_df.sort_values(['trip_id', 'stop_sequence'])
    
    st_df['next_stop_id'] = st_df.groupby('trip_id')['stop_id'].shift(-1)
    st_df['next_arrival_sec'] = st_df.groupby('trip_id')['arrival_sec'].shift(-1)
    st_df['travel_time'] = st_df['next_arrival_sec'] - st_df['arrival_sec']
    
    edges_df = st_df.dropna(subset=['next_stop_id', 'travel_time'])
    edges_df = edges_df[edges_df['travel_time'] > 0]
    
    static_edges = edges_df.groupby(['stop_id', 'next_stop_id'])['travel_time'].mean().reset_index()
    
    edge_count = len(static_edges)
    
    del st_df
    del edges_df
    gc.collect()
    
    return static_edges

def main():
    logger.info("================ CONFIGURATION ================")
    logger.info(f"Epochs:        {config.EPOCHS}")
    logger.info(f"Batch Size:    {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"Model Dir:     {config.MODEL_DIR}")
    logger.info(f"Device:        {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("===============================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    static_edges = build_static_graph_and_cleanup()

    logger.info("Loading Training data...")
    try:
        df_train = pd.read_csv(config.TRAIN_DATA_PATH, low_memory=False)
        #df_train = pd.read_csv(config.PROCESSED_DATA_PATH)
        logger.info(f"Training data loaded. Shape: {df_train.shape}")
    except FileNotFoundError:
        logger.error(f"File not found: {config.TRAIN_DATA_PATH}")
        return


    all_stop_ids = pd.concat([
        static_edges['stop_id'].astype(str), 
        static_edges['next_stop_id'].astype(str),
        df_train['stop_id'].astype(str)
    ]).unique()

    le = LabelEncoder()
    le.fit(all_stop_ids)
    
    src = le.transform(static_edges['stop_id'].astype(str))
    dst = le.transform(static_edges['next_stop_id'].astype(str))
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long).to(device)
    edge_weight = torch.tensor(static_edges['travel_time'].values, dtype=torch.float).to(device)

    node_pos_df = df_train.groupby('stop_id')[['lat', 'lon']].mean().reindex(le.classes_).fillna(0)
    x = torch.tensor(node_pos_df.values, dtype=torch.float).to(device)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(edge_index.cpu(), os.path.join(config.MODEL_DIR, 'edge_index.pt'))
    torch.save(edge_weight.cpu(), os.path.join(config.MODEL_DIR, 'edge_weight.pt'))
    torch.save(x.cpu(), os.path.join(config.MODEL_DIR, 'node_features.pt'))

    df_train['stop_idx'] = le.transform(df_train['stop_id'].astype(str))
    feature_cols = ['delay_seconds', 'speed', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'heading']
    
    X_dynamic = torch.tensor(df_train[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df_train['final_delay'].values, dtype=torch.float).view(-1, 1)
    stop_indices = torch.tensor(df_train['stop_idx'].values, dtype=torch.long)

    split_idx = int(len(df_train) * 0.8)
    logger.info("=================== BASELINE ==================")
    logger.info("--- Training Baseline: Linear Regression  ---")

    X_train_np = X_dynamic[:split_idx].numpy()
    y_train_np = y[:split_idx].numpy()

    X_val_np = X_dynamic[split_idx:].numpy()
    y_val_np = y[split_idx:].numpy()

    reg = LinearRegression()
    reg.fit(X_train_np, y_train_np)
    
    lr_model_path = os.path.join(config.MODEL_DIR, 'linear_regression.joblib')
    joblib.dump(reg, lr_model_path)
    logger.info(f"Linear Regression model saved to: {lr_model_path}")

    y_pred_train = reg.predict(X_train_np)
    mae_train = mean_absolute_error(y_train_np, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train_np, y_pred_train))

    y_pred_val = reg.predict(X_val_np)
    mae_val = mean_absolute_error(y_val_np, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val_np, y_pred_val))

    lr_model_path = os.path.join(config.MODEL_DIR, 'linear_regression.joblib')
    joblib.dump(reg, lr_model_path)

    del X_train_np, y_train_np, X_val_np, y_val_np
    del y_pred_train, y_pred_val

    logger.info(f"LR Baseline Results [TRAIN] -> MAE: {mae_train/60:.2f} min | RMSE: {rmse_train/60:.2f} min")
    # TensorDataset és DataLoader
    train_dataset = TensorDataset(X_dynamic[:split_idx], y[:split_idx], stop_indices[:split_idx])
    test_dataset = TensorDataset(X_dynamic[split_idx:], y[split_idx:], stop_indices[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. LOGGING: Model Architecture
    model = BKK_GNN_Weighted(num_node_features=2, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.L1Loss()

    trainable, total_params = count_parameters(model)
    logger.info("================ MODEL ARCHITECTURE ================")
    logger.info(str(model))
    logger.info(f"Total Parameters:     {total_params}")
    logger.info(f"Trainable Parameters: {trainable}")
    logger.info("====================================================")

    logger.info("Starting training loop...")
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    
    for epoch in range(1, config.EPOCHS + 1):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y, batch_idx in train_loader:
            batch_X, batch_y, batch_idx = batch_X.to(device), batch_y.to(device), batch_idx.to(device)
            
            optimizer.zero_grad()
            out = model(x, edge_index, edge_weight, batch_X, batch_idx)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_idx in test_loader:
                batch_X, batch_y, batch_idx = batch_X.to(device), batch_y.to(device), batch_idx.to(device)
                out = model(x, edge_index, edge_weight, batch_X, batch_idx)
                all_preds.append(out.cpu())
                all_targets.append(batch_y.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        val_mae = mean_absolute_error(all_targets, all_preds)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        elapsed = time.time() - start_time
        
        logger.info(f"Epoch {epoch}/{config.EPOCHS} | Time: {elapsed:.1f}s | "
                    f"Train Loss: {avg_train_loss/60:.2f} | "
                    f"Val MAE: {val_mae/60:.2f} min | Val RMSE: {val_rmse/60:.2f} min")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_gnn_model.pth'))
            joblib.dump(le, os.path.join(config.MODEL_DIR, 'label_encoder.pkl'))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
    logger.info("==================== EVALUATION ====================")
    logger.info("Training complete. Best model saved.")
    logger.info(f"LR Validation MAE:   {mae_val/60:.2f} min")
    logger.info(f"LR Validation RMSE:  {rmse_val/60:.2f} min")
    logger.info(f"GNN Validation MAE:  {best_val_mae/60:.2f} min")
    logger.info(f"GNN Validation RMSE: {best_val_rmse/60:.2f} min")



if __name__ == "__main__":
    main()