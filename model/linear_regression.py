from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def run_linear_regression(df, preprocessor, *, target="salario", test_size=0.3, seed=42):
    """
    Executa uma regressão linear simples como instrumento de medição de utilidade.

    Responsabilidades:
    - realizar split controlado
    - treinar modelo determinístico
    - gerar predições

    Não calcula métricas.
    Não conhece ε.
    Não participa de decisões experimentais.
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

    y_pred = model.predict(X_test)

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "n_features": X_train.shape[1],
        "model": model,
    }
