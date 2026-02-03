from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_random_forest(df, preprocessor, *, target="salario", test_size=0.3, seed=42):

    """
    Executa um Random Forest Regressor como instrumento de medição de utilidade 
    e de potencial memorização para análise de risco de inferência.

    Responsabilidades:
    - realizar split controlado e reproduzível
    - treinar modelo de maior capacidade que o linear
    - gerar predições para treino e teste
    - permitir medir diferença de comportamento entre amostras vistas e não vistas

    Papel no experimento:
    - servir como contraste ao modelo linear
    - expor efeitos de memorização que podem aumentar risco de MIA
    - avaliar como a DP afeta modelos de maior capacidade

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

    model = RandomForestRegressor(n_estimators= 100, max_depth= None, min_samples_leaf=1, random_state= seed,n_jobs=-1)
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