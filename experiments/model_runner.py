from metrics import compute_utility_metrics
from preprocessing import build_preprocessor
from sklearn.model_selection import train_test_split


def _attach_audit_data(prediction_result, df, *, target_col, seed, test_size):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    prediction_result.X_train = X_train
    prediction_result.X_test = X_test
    prediction_result.train_indices = X_train.index
    prediction_result.test_indices = X_test.index
    prediction_result.target_col = target_col
    prediction_result.seed = seed
    prediction_result.test_size = test_size

    return prediction_result


def run_model(df, model_runner, *, seed=42, test_size=0.3, target_col="salario"):

    preprocessor = build_preprocessor(df=df) #Esse

    model_output = model_runner(
        df=df,
        preprocessor=preprocessor,
        seed=seed,
        test_size=test_size,
    )

    model_output = _attach_audit_data(
        prediction_result=model_output,
        df=df,
        target_col=target_col,
        seed=seed,
        test_size=test_size,
    )

    return compute_utility_metrics(prediction_result= model_output), model_output
