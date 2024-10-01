from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pickle

from rio_airbnb.config import MODELS_DIR, PROCESSED_DATA_DIR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error # type: ignore
import skopt

app = typer.Typer()

def load_mais_recente(model_prefix="glm_model"):
    """Carrega o modelo mais recente salvo no diretório MODEL_DIR."""
    model_files = list(MODELS_DIR.glob(f"{model_prefix}_*.pkl"))
    
    if not model_files:
        raise FileNotFoundError("Nenhum modelo encontrado no diretório.")

    # Ordena os arquivos pelo timestamp de modificação (mais recente primeiro)
    most_recent_model = max(model_files, key=lambda f: f.stat().st_mtime)
    
    # Carrega o modelo
    with open(most_recent_model, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    logger.info(f"Modelo mais recente carregado: {most_recent_model}")

    # Verifica se o modelo é uma instância de BayesSearchCV
    if isinstance(loaded_model, skopt.searchcv.BayesSearchCV):
        logger.info("O modelo é uma instância de BayesSearchCV, retornando best_estimator_")
        return loaded_model.best_estimator_ # type: ignore
    
    return loaded_model

def print_model_metrics(model, X_test, y_test):
    """Imprime as métricas de desempenho do modelo fornecido."""
    y_pred = model.predict(X_test)

    # Calcular as métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Exibir as métricas
    print("[MSE] Mean Squared Error do modelo baseline: ", mse)
    print("[RMSE] Root Mean Squared Error do modelo baseline: ", rmse)
    print("[MAE] Mean Absolute Error do modelo baseline: ", mae)
    print("[MAPE] Mean Absolute Percentage Error do modelo baseline: ", mape)
    print("[R2] R2 Score do modelo baseline: ", r2)

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
