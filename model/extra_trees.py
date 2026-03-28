from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


def run_extra_trees(df, preprocessor, *, target="salario", test_size=0.3, seed=42):
    """
    Executa o modelo Extra Trees Regressor.

    Papel no experimento:
    - variaÃ§Ã£o do Random Forest com maior aleatoriedade
    - ajuda a verificar se estabilidade do RF se mantÃ©m com splits mais randÃ´micos
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

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=3,
        max_features=0.5,
        random_state=seed
    )

    model.fit(X_train, y_train)
    model.preprocessor_ = preprocessor

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    return {
        "y_train_true": y_train,
        "y_train_pred": y_train_pred,
        "y_test_true": y_test,
        "y_test_pred": y_test_pred,
        "model": model,
    }
