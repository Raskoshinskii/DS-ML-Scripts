def prepare_lstm_data(seq_len, series):
    """
    Prepares required data format for LSTM (sequence).
    
    Parametrs
    ---------
    seq_len: int
        Sequence length.
    series: pd.Series
        Time Series data.
        
    Returns
    -------
    tuple
        Train data (LSTM tensor) and target variable.
    """
    x_values = []
    y_values = []
    
    for i in range(seq_len, series.shape[0]):
        x_values.append(series[i - seq_len:i])
        y_values.append(series[i])
        
    X, y = np.array(x_values), np.array(y_values)
    tensor_shape = (X.shape[0], seq_len, 1)
    X = np.reshape(X, tensor_shape)
    return X, y


def get_lstm_cv_score(
    model,
    series,
    n_splits,
    scaler,
    cv_metric,
    inverse_transform=True,
    skip_first_fold_error=False
):
    """
    Runs Time Series Cross-Validation.
    
    Parameters
    ----------
    model: callable
        Testing Model.
    series: pd.Series
        Time Series data.
    n_splits: int
        TimeSeriesCV splits.
    scaler: callable
        Scaler that should be used.
    cv_metric: callable
        Metric to use (e.g MAPE, MSE, ...).
    inverse_transform: bool
        Wether to apply inverse scaling operation.
    skip_first_fold_error: bool
        If the first error fold should be ommted.
        
    Returns
    -------
    np.array
        Array with CV errors.
    """
    errors = []
    values = series.values
    
    ts_cv = TimeSeriesSplit(n_splits=n_splits)
    for train, test in ts_cv.split(values):
        train_data = values[train]
        test_data = values[test]
        
        train_scaled = scaler.fit_transform(pd.DataFrame(train_data))
        test_scaled = scaler.transform(pd.DataFrame(test_data))
        
        # LSTM data prep
        X_train, y_train = prepare_lstm_data(seq_len, series=train_scaled)
        X_test, y_test = prepare_lstm_data(seq_len, series=test_scaled)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE
        )
        if inverse_transform:
            preds = scaler.inverse_transform(model.predict(X_test))
            y_true = scaler.inverse_transform(y_test)
            error = cv_metric(y_true, preds)
            errors.append(error)
        else:
            preds = model.predict(X_test)
            error = cv_metric(y_test, preds)
            errors.append(error)
        print('Fold Fitted!')  
        
    print('Folds Errors: ', errors)
    if skip_first_fold_error:
        return np.mean(np.array(errors[1:]))
    return np.mean(np.array(errors))


def get_lstm_forecast(
    train_df,
    fitted_model,
    seq_len,
    n_steps,
    last_date
):
    """
    Provides LSTM forecast according to n_steps value (i.e. prediction horizon).
    
    Parameters
    ----------
    train_df: pd.DataFrame
        Time Series data (all available data).
    fitted_model: callable
        Fitted model.        
    seq_len: int
        Sequence length.
    n_steps: int
        Prediction horizon.
    last_date: pd.DatetimeIndex
        Last datetime index from train_df.
        
    Returns
    -------
    pd.DataFrame
        Forecast DataFrame.
    """
    # 1. Получение прогнозных значений согласно n_steps (горизонт прогнозирования)
    target_sequence = train_df[-seq_len:] # последовательность, необходимая для прогноза LSTM
    
    for _ in range(n_steps):
        X_test = target_sequence[-seq_len:] # последние n значений необходимые для предсказания 
        X_test = X_test.reshape((1, seq_len, 1)) # необходимый формат для LSTM
        model_pred = model.predict(X_test)[0][0]
        target_sequence = np.append(target_sequence, model_pred)
        
    # 2. Получение дат для прогнозных значений
    data_frea = last_date.freq
    prediction_dates = pd.date_range(
        last_date,
        periods=n_steps+1,
        freq=data_freq
    )[1:]
    
    # 3. Итоговый DataFrame
    target_sequence = target_sequence[seq_len:]
    pred_df = pd.DataFrame(
        data=target_sequence,
        index=prediction_dates,
        columns=['forecast']
    )
    return pred_df