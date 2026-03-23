
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_models(X, y_conv, y_def):
    X_train, X_test, y_train, y_test = train_test_split(X, y_conv, test_size=0.2)
    model_conv = RandomForestClassifier().fit(X_train, y_train)
    preds = model_conv.predict(X_test)

    model_def = RandomForestClassifier().fit(X, y_def)

    metrics = {"accuracy": accuracy_score(y_test, preds)}
    fi = model_conv.feature_importances_
    return model_conv, model_def, metrics, fi

def predict_user(user, model_conv, model_def):
    import pandas as pd
    df = pd.DataFrame([user])
    return model_conv.predict(df)[0], model_def.predict(df)[0]
