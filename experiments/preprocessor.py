from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor():

    categorical = ["cargo", "setor"]
    numerical = ["idade", "tempo_na_empresa", "nota_media"]

    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), categorical),
        ("num", "passthrough", numerical),
    ])
