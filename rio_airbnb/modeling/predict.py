from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pickle

from rio_airbnb.config import MODELS_DIR, PROCESSED_DATA_DIR

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
    return loaded_model


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
