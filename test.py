

    data = load_and_merge_data()
    if not data.empty:
        print(f"Loaded data shape: {data.shape}")
        analyze_correlations(data.copy())

    rf_results = tune_and_evaluate_rf(data.copy())

    # --- 3. Advanced RNN Pipeline (Tuned) ---
    print("\n--- Tuning and Evaluating Advanced RNN (LSTM/GRU + Attention) ---")


    # Run correlation analysis
    analyze_correlations(data.copy())

    # --- 2. Random Forest Pipeline (Tuned) ---
    rf_results = tune_and_evaluate_rf(data.copy())

    # --- 3. Advanced RNN Pipeline (Tuned) ---
    print("\n--- Tuning and Evaluating Advanced RNN (LSTM/GRU + Attention) ---")

    # 3a. Prepare data (full dataset, pre-sequencing)
    df_lstm, features, target, scaler_X, scaler_y = preprocess_data_lstm(data.copy())
    N_FEATURES = len(features)

    # 3b. Instantiate the HyperModel
    hypermodel = CropYieldHyperModel(
        full_data=df_lstm,
        feature_cols=features,
        target_col=target,
        n_features=N_FEATURES
    )

    # 3c. Instantiate the Tuner (Hyperband) [21, 25]
    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective('val_loss', direction='min'),
        max_epochs=50,
        factor=3,
        directory='keras_tuner_dir',
        project_name='crop_yield_tuning',
        overwrite=True
    )

    # 3d. Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 3e. Run the search
    # The 'fit' method of CropYieldHyperModel will be called
    # with the 'None, None' arguments, which we ignore.
    tuner.search(
        callbacks=[early_stop]
    )

    # 3f. Get best hyperparameters and retrain the final model
    print("\n--- Training Final RNN Model ---")
    tuner.results_summary()

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best RNN Hyperparameters: {best_hps.values}")

    # Build the final model with these HPs
    final_rnn_model = hypermodel.build(best_hps)

    # Now, retrain this single best model on the *full* dataset
    # (or a final train/test split). We will create the final
    # train/test split using the *optimal* seq_len.

    optimal_seq_len = best_hps.get('seq_len')
    print(f"Retraining final model with optimal seq_len: {optimal_seq_len}")

    X_seq_final, y_seq_final = create_grouped_sequences(
        df_lstm, features, target, optimal_seq_len
    )

    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_seq_final, y_seq_final, test_size=0.2, random_state=42
    )

    # Re-train on the full training set for a robust number of epochs [29]
    final_rnn_model.fit(
        X_train_final,
        y_train_final,
        epochs=100, # Train longer on the final model
        batch_size=32,
        validation_data=(X_test_final, y_test_final),
        callbacks=[early_stop],
        verbose=1
    )

    # 3g. Evaluate the final, retrained model
    rnn_results = evaluate_final_model(
        final_rnn_model, X_test_final, y_test_final, scaler_y
    )

    # --- 4. Compare Models ---
    print("\n--- Final Model Comparison ---")

    # For a complete comparison, we'll add the *original* baseline results
    # (Note: These are hard-coded placeholders for demonstration)
    baseline_rf = {'Model': 'Random Forest (Baseline)', 'R2': 0.95, 'RMSE': 25000, 'MAE': 10000}
    baseline_lstm = {'Model': 'LSTM (Baseline)', 'R2': 0.80, 'RMSE': 50000, 'MAE': 25000}

    results_df = pd.DataFrame([
        baseline_rf,
        baseline_lstm,
        rf_results,
        rnn_results
    ])
    results_df = results_df.set_index('Model')

    print(results_df.to_string())