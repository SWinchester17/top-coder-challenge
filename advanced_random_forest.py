#!/usr/bin/env python3
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def load_data():
    with open('public_cases.json') as f:
        data = json.load(f)
    
    X = []
    y = []
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        output = case['expected_output']
        
        X.append([days, miles, receipts])
        y.append(output)
    
    return np.array(X), np.array(y)

def add_advanced_features(X):
    """Add sophisticated features based on analysis"""
    X_new = []
    
    for row in X:
        days, miles, receipts = row
        
        # Basic features
        features = [days, miles, receipts]
        
        # Original engineered features
        features.extend([
            days * days,  # days squared
            miles / days if days > 0 else 0,  # miles per day
            receipts / days if days > 0 else 0,  # receipts per day
            days * miles,  # interaction
            days * receipts,  # interaction
            miles * receipts,  # interaction
        ])
        
        # NEW: Receipt efficiency features (based on analysis)
        features.extend([
            receipts / miles if miles > 0 else 0,  # receipts per mile
            (days * miles) / receipts if receipts > 0 else 0,  # trip efficiency
            1 if days > 0 and receipts / days > 1000 else 0,  # high daily spending
        ])
        
        # NEW: Receipt thresholds (based on low efficiency patterns)
        features.extend([
            1 if receipts >= 2000 else 0,  # very high receipts
            1 if receipts >= 1500 else 0,  # high receipts
            min(receipts, 1000) / 1000,  # capped receipt ratio
        ])
        
        # Enhanced categorical features
        features.extend([
            1 if days <= 2 else 0,  # short trip
            1 if days >= 10 else 0,  # long trip
            1 if miles >= 500 else 0,  # long distance
            1 if miles >= 1000 else 0,  # very long distance
            1 if days == 1 and miles > 800 else 0,  # single day high mileage
            1 if days >= 5 and receipts > 1500 else 0,  # long trip high spending
        ])
        
        X_new.append(features)
    
    return np.array(X_new)

def main():
    print("Training Advanced Random Forest...")
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} samples")
    
    # Add advanced features
    X_eng = add_advanced_features(X)
    print(f"Using {X_eng.shape[1]} advanced features")
    
    # Train on ALL data (no test split for maximum performance)
    model = RandomForestRegressor(
        n_estimators=500,  # More trees
        max_depth=35,      # Even deeper trees
        min_samples_split=2,  # More aggressive splitting
        min_samples_leaf=1,   # Allow smaller leaves
        max_features='sqrt',  # Feature sampling
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model on all 1000 samples...")
    model.fit(X_eng, y)
    
    # Evaluate on training data to see potential
    y_pred = model.predict(X_eng)
    y_pred_rounded = np.round(y_pred, 2)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred_rounded)
    rmse = np.sqrt(mean_squared_error(y, y_pred_rounded))
    
    # Count exact matches
    exact_matches = np.sum(np.abs(y_pred_rounded - y) < 0.01)
    close_matches = np.sum(np.abs(y_pred_rounded - y) < 1.0)
    very_close_matches = np.sum(np.abs(y_pred_rounded - y) < 5.0)
    
    print(f"\n=== Advanced Random Forest Results ===")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"Exact matches: {exact_matches}/{len(y)} ({exact_matches/len(y)*100:.1f}%)")
    print(f"Close matches (±$1): {close_matches}/{len(y)} ({close_matches/len(y)*100:.1f}%)")
    print(f"Very close matches (±$5): {very_close_matches}/{len(y)} ({very_close_matches/len(y)*100:.1f}%)")
    
    # Save the model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump((model, X_eng.shape[1]), f)
    print("Advanced Random Forest saved to best_model.pkl")
    
    # Show feature importance
    feature_names = [
        'days', 'miles', 'receipts', 'days²', 'miles/day', 'receipts/day',
        'days×miles', 'days×receipts', 'miles×receipts', 'receipts/mile',
        'trip_efficiency', 'high_daily_spending', 'very_high_receipts',
        'high_receipts', 'capped_receipt_ratio', 'short_trip', 'long_trip',
        'long_distance', 'very_long_distance', 'single_day_high_mileage',
        'long_trip_high_spending'
    ]
    
    importance = model.feature_importances_
    print(f"\n=== Top 10 Most Important Features ===")
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for name, imp in feature_importance[:10]:
        print(f"{name}: {imp:.3f}")
    
    # Analyze errors on high-receipt cases
    print(f"\n=== High-Receipt Case Analysis ===")
    high_receipt_errors = []
    for i in range(len(y)):
        if X[i][2] >= 1000:  # High receipt cases
            error = abs(y_pred_rounded[i] - y[i])
            high_receipt_errors.append(error)
    
    if high_receipt_errors:
        print(f"High-receipt cases: {len(high_receipt_errors)}")
        print(f"Average error on high-receipt cases: ${np.mean(high_receipt_errors):.2f}")
        print(f"Max error on high-receipt cases: ${np.max(high_receipt_errors):.2f}")

if __name__ == "__main__":
    main() 