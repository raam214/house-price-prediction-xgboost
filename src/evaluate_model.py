from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model using regression metrics.
    """
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return rmse, r2
