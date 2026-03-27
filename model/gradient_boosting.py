from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def run_gradient_boosting(df, preprocessor, *, target="salario", test_size=0.3, seed=42):
    """
    Executa o modelo Gradient Boosting Regressor.

    Papel no experimento:
    - comparativo direto com XGBoost (boosting baseado em Ã¡rvores)
    - ajuda a separar efeito de boosting vs implementaÃ§Ã£o especÃ­fica
    """

    if target not in df.columns:
        raise ValueError(f"Target '{target}' nÃ£o encontrado no dataset.")

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

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=seed,
    )

    model.fit(X_train, y_train)
    model.preprocessor_ = preprocessor

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_true": y_train,
        "y_train_pred": y_train_pred,
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "n_features": X_train.shape[1],
        "model": model,
    }
