from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from rio_airbnb.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

import pickle
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
from datetime import datetime

class GLMForwardSelector:
    """Classe para realizar a seleção stepwise (forward) para modelos GLM usando AIC como critério."""
    
    def __init__(self, data, response, family, model_prefix="glm_model"):
        """
        Inicializa o seletor com os dados, a variável resposta e a família de distribuição.
        
        :param data: DataFrame com os dados de entrada.
        :param response: String com o nome da variável resposta.
        :param family: Família de distribuição a ser usada no GLM.
        :param model_prefix: Prefixo para o nome do arquivo de modelo (padrão: "glm_model").
        """
        self.data = data
        self.response = response
        self.family = family
        self.selected = []
        self.remaining = set(data.columns)
        self.remaining.remove(response)
        self.current_score = float('inf')
        self.best_new_score = float('inf')
        self.model = None
        self.model_prefix = model_prefix

    def _get_formula(self, candidate):
        """Gera a fórmula do modelo com a variável candidata incluída."""
        return "{} ~ {} + 1".format(self.response, ' + '.join(self.selected + [candidate]))

    def _fit_model(self, formula):
        """Ajusta o modelo GLM com a fórmula fornecida."""
        return smf.glm(formula, self.data, family=self.family).fit()

    def _save_model(self):
        """Salva o modelo no caminho fornecido em MODELS_DIR, com timestamp."""
        # Gera o timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Adiciona o timestamp ao nome do arquivo
        model_filename = f"{self.model_prefix}_{timestamp}.pkl"
        save_path = MODELS_DIR / model_filename
        
        # Salva o modelo
        with open(save_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        logger.info(f"Modelo salvo em: {save_path}")

    def select(self):
        """Executa o processo de seleção stepwise (forward) usando o critério AIC."""
        while self.remaining and self.current_score == self.best_new_score:
            scores_with_candidates = []
            
            for candidate in self.remaining:
                formula = self._get_formula(candidate)
                model = self._fit_model(formula)
                score = model.aic
                scores_with_candidates.append((score, candidate))
            
            scores_with_candidates.sort(reverse=True)
            self.best_new_score, best_candidate = scores_with_candidates.pop()
            
            if self.current_score > self.best_new_score:
                self.remaining.remove(best_candidate)
                self.selected.append(best_candidate)
                self.current_score = self.best_new_score

        # Ajusta o modelo final com as variáveis selecionadas
        final_formula = "{} ~ {} + 1".format(self.response, ' + '.join(self.selected))
        self.model = self._fit_model(final_formula)

        # Salva o modelo automaticamente após o processo de seleção
        self._save_model()
        
        return self.model

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
