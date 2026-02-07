from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def run_linear_regression(df, preprocessor, *, target="salario", test_size=0.3, seed=42):
    """
    Executa uma regressão linear simples.

    Responsabilidades:
    - realizar split controlado
    - treinar modelo determinístico
    - gerar predições

    """

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado no dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    y_train_pred = model.predict(X_train)
    return {
        "X_train" : X_train,
        "X_test" : X_test,
        "y_train_true": y_train,
        "y_train_pred": y_train_pred,
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "n_features": X_train.shape[1],
        "model": model,
    }
