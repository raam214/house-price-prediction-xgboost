import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def train_model(X, y):
    """
    Train XGBoost regression model for house price prediction.
    """
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # initialize model with basic parameters
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # train model
    model.fit(X_train, y_train)

    # evaluate model
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)

    print(f"Model R2 score: {score:.2f}")

    # save trained model
    joblib.dump(model, "model/xgboost_model.pkl")

    return model
