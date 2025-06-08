#!/usr/bin/env python3
import sys
import pickle
import numpy as np

def add_advanced_features(days, miles, receipts):
    """Add advanced engineered features for ML model (matching improved_model.py)"""
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
    
    return np.array(features).reshape(1, -1)

def calculate_reimbursement(days, miles, receipts):
    """Use ML model for reimbursement calculation"""
    days = int(days)
    miles = float(miles)
    receipts = float(receipts)
    
    try:
        # Load the trained model
        with open('best_model.pkl', 'rb') as f:
            model, expected_features = pickle.load(f)
        
        # Prepare features using advanced feature engineering
        X = add_advanced_features(days, miles, receipts)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return round(prediction, 2)
        
    except FileNotFoundError:
        # Fallback to simple linear formula if model not found
        print("Warning: ML model not found, using fallback linear formula", file=sys.stderr)
        return round(75 * days + 0.5 * miles + 0.5 * receipts, 2)
    except Exception as e:
        print(f"Error using ML model: {e}", file=sys.stderr)
        return round(75 * days + 0.5 * miles + 0.5 * receipts, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ml_calculate.py <days> <miles> <receipts>")
        sys.exit(1)
    
    days = sys.argv[1]
    miles = sys.argv[2]
    receipts = sys.argv[3]
    
    result = calculate_reimbursement(days, miles, receipts)
    print(result) 