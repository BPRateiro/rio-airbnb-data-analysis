from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from rio_airbnb.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

import pickle
from pathlib import Path
import statsmodels.formula.api as smf
from datetime import datetime

from skopt import BayesSearchCV
from xgboost import XGBRegressor

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

class BayesianXGBoostOptimizer:
    """Classe para realizar a otimização de hiperparâmetros do XGBoost usando busca bayesiana."""

    def __init__(self, model_prefix="xgb_model", n_iter=50, random_state=75):
        """
        Inicializa o otimizador com o prefixo do modelo, número de iterações e seed.

        :param model_prefix: Prefixo para o nome do arquivo de modelo (padrão: "xgb_model").
        :param n_iter: Número de iterações para a busca bayesiana (padrão: 50).
        :param random_state: Semente aleatória para reprodutibilidade (padrão: 75).
        """
        self.model_prefix = model_prefix
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None

        # Definir o espaço de busca
        self.search_spaces = {
            'n_estimators': (50, 500),  # Número de estimadores
            'max_depth': (3, 10),       # Profundidade máxima da árvore
            'min_child_weight': (1, 6), # Peso mínimo da criança
            'gamma': (0, 0.5),          # Parâmetro gamma
            'subsample': (0.6, 0.9),    # Subamostragem
            'colsample_bytree': (0.6, 0.9), # Fração de colunas usadas por árvore
            'reg_alpha': (1e-5, 100, 'log-uniform') # Regularização L1
        }

    def _save_model(self):
        """Salva o modelo em um arquivo com um timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_prefix}_{timestamp}.pkl"
        save_path = MODELS_DIR / model_filename
        
        # Salvar o modelo
        with open(save_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        
        logger.info(f"Modelo salvo em: {save_path}")

    def fit(self, X_train, y_train, cv=5):
        """Executa a otimização bayesiana e ajusta o modelo."""
        # Definir o modelo base XGBoost
        xgb = XGBRegressor(
            learning_rate=0.1,
            seed=self.random_state,
            eval_metric="mape"
        )

        # Configurar o BayesSearchCV com verbose=0
        self.model = BayesSearchCV(
            estimator=xgb,
            search_spaces=self.search_spaces,
            n_iter=self.n_iter,
            cv=cv,
            n_jobs=-1,
            scoring='neg_mean_absolute_percentage_error',
            random_state=self.random_state,
            verbose=0  # Desativar o verbose
        )

        logger.info("Iniciando o processo de ajuste do modelo...")
        # Realizar o ajuste
        self.model.fit(X_train, y_train)
        logger.info("Ajuste do modelo concluído.")

        # Salvar o melhor modelo encontrado
        self._save_model()

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
