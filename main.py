# main.py
from datetime import date

import polars as pl

from umabozu.data_preprocessing.data_loader import load_data_from_db
from umabozu.data_preprocessing.feature_engineer import (
    generate_features,
    get_feature_cols,
)
from umabozu.data_preprocessing.preprocessor import clean_data
from umabozu.database.crud import create_tables
from umabozu.models.predictor import predict_probabilities, save_model
from umabozu.models.trainer import train_model
from umabozu.simulation.bet_simulator import calculate_payout, simulate_betting
from umabozu.simulation.odds_calculator import calculate_expected_value


def main():
    print("--- Starting Umabozu AI Prediction System (Refined Baseline) ---")

    # 0. Ensure Database Tables Exist
    create_tables()

    # 1. Skip scraping (Already collected)
    print("\n[Step 1/8] Skipping scraping...")

    # 2. Load Data from DB
    print("\n[Step 2/8] Loading data from database...")
    df = load_data_from_db()
    if df.is_empty():
        print("Error: No data found in database.")
        return
    print(f"Data loaded: {len(df)} rows.")

    # 3. Clean Data
    print("\n[Step 3/8] Cleaning data...")
    cleaned_df = clean_data(df)
    print("Data cleaning complete.")

    # 4. Feature Engineering
    print("\n[Step 4/8] Generating features...")
    featured_df = generate_features(cleaned_df)
    print("Feature engineering complete.")

    # Get consistent feature columns
    actual_feature_cols = [
        col for col in get_feature_cols() if col in featured_df.columns
    ]

    # 5. Train Model (Using Default Baseline)
    print("\n[Step 5/8] Training LightGBM model (Default Parameters)...")
    featured_df = featured_df.sort("race_date")
    split_idx = int(len(featured_df) * 0.8)
    train_df = featured_df.head(split_idx)
    test_df = featured_df.tail(len(featured_df) - split_idx)

    print(f"Training on {len(train_df)} samples, Testing on {len(test_df)} samples.")

    # optimize=False to keep the good baseline behavior
    model = train_model(train_df, optimize=False)
    save_model(model, "trained_models/lgbm_model.pkl")
    print("Model trained.")

    # 6. Predict Probabilities
    print("\n[Step 6/8] Predicting probabilities on test set...")
    predicted_probas_df = predict_probabilities(model, test_df, actual_feature_cols)
    print("Probabilities predicted.")

    # 7. Calculate Expected Value
    print("\n[Step 7/8] Calculating expected values...")
    df_with_ev = calculate_expected_value(predicted_probas_df, test_df)
    print("Expected values calculated.")

    # 8. Simulate Betting and Calculate Payout
    print("\n[Step 8/8] Simulating betting with Noise Filter...")
    # Updated thresholds found via strategy optimization for best ROI
    selected_bets = simulate_betting(df_with_ev, ev_threshold=1.1, min_probability=0.13)

    # actual_results_df = test_df.select(["race_id", "horse_id", "rank"])
    investment, total_payout, roi = calculate_payout(selected_bets)

    print("\n--- Final Simulation Results (Refined Baseline) ---")
    print(f"Number of total bets considered: {len(df_with_ev)}")
    print(f"Number of selected bets: {len(selected_bets)}")
    if not selected_bets.is_empty():
        print(f"Total investment: {investment:.2f} JPY")
        print(f"Total payout: {total_payout:.2f} JPY")
        print(f"Return on Investment (ROI): {roi:.2f}%")
    else:
        print("No bets were placed.")
    print("\n--- Umabozu AI Prediction System Finished ---")


if __name__ == "__main__":
    main()
