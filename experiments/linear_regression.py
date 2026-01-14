from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run_linear_regression(df, preprocessor):

    X = df.drop(columns=["salario"])
    y = df["salario"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error( y_true= y_test, y_pred= y_pred) ** 0.5


    return rmse, y_test, y_pred
