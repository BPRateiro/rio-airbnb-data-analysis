from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from rio_airbnb.config import PROCESSED_DATA_DIR

app = typer.Typer()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

import pandas as pd
import ast

class PreProcessamento(BaseEstimator, TransformerMixin):
    def __init__(self, col_referencia='last_scraped', col_datas=None, col_descricoes=None, col_preco='price', col_comodidades='amenities', col_verificacoes='host_verifications', col_resposta='host_response_time', col_booleana=None, col_percentual=None, col_room_type='room_type', limite_categorias=0.05, colunas_para_remover=None):
        self.col_referencia = col_referencia
        self.col_datas = col_datas or ['last_scraped', 'first_review', 'last_review', 'host_since']
        self.col_descricoes = col_descricoes or ['description', 'neighborhood_overview', 'host_about']
        self.col_preco = col_preco
        self.col_comodidades = col_comodidades
        self.col_verificacoes = col_verificacoes
        self.col_resposta = col_resposta
        self.col_booleana = col_booleana or ['host_has_profile_pic', 'host_identity_verified', 'host_is_superhost', 'instant_bookable']
        self.col_percentual = col_percentual or ['host_acceptance_rate', 'host_response_rate']
        self.col_room_type = col_room_type
        self.limite_categorias = limite_categorias
        self.colunas_para_remover = colunas_para_remover or [
            'id', 'scrape_id', 'host_id', 'host_name', 'source', 'name', 
            'picture_url', 'listing_url', 'host_url', 'host_thumbnail_url', 'host_picture_url',
            'calendar_last_scraped', 'bathrooms_text', 'has_availability',
            'host_has_profile_pic', 'verification_phone', 'host_location',
            'neighbourhood', 'neighbourhood_group_cleansed', 'calendar_updated', 'license',
            'last_scraped', 'first_review', 'last_review', 'description', 'host_since',
            'host_about', 'amenities', 'neighborhood_overview', 'host_verifications'
        ]

    def _converter_datas_para_dias(self, X):
        """Converte datas para intervalo de dias."""
        X[self.col_referencia] = pd.to_datetime(X[self.col_referencia], errors='coerce')
        for data in self.col_datas:
            X[data] = pd.to_datetime(X[data], errors='coerce')
            if data != self.col_referencia:
                X[f'days_since_{data}'] = (X[self.col_referencia] - X[data]).dt.days
        X['was_reviewed'] = X['first_review'].notna()
        X = X.rename(columns={'days_since_host_since': 'days_since_host_active'})
        return X

    def _converter_descricoes_para_tamanho(self, X):
        """Substitui descrições por contagem de caracteres."""
        for descricao in self.col_descricoes:
            X[f'{descricao}_length'] = X[descricao].fillna('').apply(len)
        return X

    def _limpar_e_converter_preco(self, X):
        """Remove símbolos de moeda e converte para float."""
        X[self.col_preco] = X[self.col_preco].replace(r'[\$,]', '', regex=True).astype(float)
        X.dropna(subset=[self.col_preco], inplace=True)
        return X

    def _contar_comodidades(self, X):
        """Substitui amenities por sua contagem."""
        X['num_amenities'] = X[self.col_comodidades].apply(ast.literal_eval).apply(len)
        return X

    def _expandir_verificacoes_host(self, X):
        """Expande a coluna host_verifications em dummies."""
        X[self.col_verificacoes] = X[self.col_verificacoes].apply(ast.literal_eval)
        X = X.join(pd.get_dummies(X.explode(self.col_verificacoes)[self.col_verificacoes], 
                                  prefix='verification').groupby(level=0).max())
        return X

    def _adicionar_flag_resposta_host(self, X):
        """Adiciona flag que indica se o host chegou a responder."""
        X['host_responded'] = X[self.col_resposta].notna()
        return X

    def _converter_colunas_booleanas(self, X):
        """Converte colunas booleanas e substitui 't'/'f' por True/False."""
        X.replace({'t': True, 'f': False}, inplace=True)
        X = X.infer_objects(copy=False)
        X[self.col_booleana] = X[self.col_booleana].astype(bool)
        return X

    def _converter_colunas_percentuais(self, X):
        """Converte colunas percentuais para float."""
        X[self.col_percentual] = X[self.col_percentual].apply(lambda x: x.str.rstrip('%').astype(float) / 100)
        return X

    def _agrupar_categorias_raras(self, X):
        """Agrupa categorias com baixa representatividade."""
        frequencias = X[self.col_room_type].value_counts(normalize=True)
        mascara = frequencias < self.limite_categorias
        X[self.col_room_type] = X[self.col_room_type].apply(lambda x: 'Others' if mascara[x] else x)
        return X

    def _remover_colunas(self, X):
        """Remove colunas desnecessárias."""
        X.drop(columns=self.colunas_para_remover, inplace=True)
        return X

    def _ordenar_colunas(self, X):
        """Ordena as colunas alfabeticamente."""
        return X[sorted(X.columns)]

    def fit(self, X, y=None):
        """Fit não realiza ajustes, mas é requerido para compatibilidade com o pipeline."""
        return self

    def transform(self, X):
        """Aplica todas as transformações ao DataFrame."""
        X_copia = X.copy()
        X_copia = self._converter_datas_para_dias(X_copia)
        X_copia = self._converter_descricoes_para_tamanho(X_copia)
        X_copia = self._limpar_e_converter_preco(X_copia)
        X_copia = self._contar_comodidades(X_copia)
        X_copia = self._expandir_verificacoes_host(X_copia)
        X_copia = self._adicionar_flag_resposta_host(X_copia)
        X_copia = self._converter_colunas_booleanas(X_copia)
        X_copia = self._converter_colunas_percentuais(X_copia)
        X_copia = self._agrupar_categorias_raras(X_copia)
        X_copia = self._remover_colunas(X_copia)
        X_copia = self._ordenar_colunas(X_copia)
        return X_copia
    
class SepararOutliersIQR(BaseEstimator, TransformerMixin):
    def __init__(self, coluna):
        """Inicializa com o nome da coluna onde será aplicado o IQR."""
        self.coluna = coluna
        self.outliers = None  # Atribuirá os outliers encontrados durante a transformação

    def fit(self, X, y=None):
        """Fit não faz nada aqui, mas é necessário para compatibilidade com o Pipeline."""
        return self

    def transform(self, X):
        """Aplica a separação de outliers baseado no IQR."""
        # Filtrar valores iguais a zero
        df = X[X[self.coluna] > 0].copy()
        
        # Cálculo do IQR
        Q1 = df[self.coluna].quantile(0.25)
        Q3 = df[self.coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # DataFrame sem outliers
        df_sem_outliers = df[(df[self.coluna] >= limite_inferior) & (df[self.coluna] <= limite_superior)]
        
        # DataFrame com outliers (armazenado na variável self.outliers)
        self.outliers = df[(df[self.coluna] < limite_inferior) | (df[self.coluna] > limite_superior)]
        
        return df_sem_outliers

    def _get_outliers(self):
        """Retorna o DataFrame com os outliers, se necessário."""
        return self.outliers

class RemoverFeaturesCorrelacionadas(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=None, salvar=False, parquet_path='../data/bronze/feature_engineering_input.parquet'):
        """Inicializa a classe com as colunas a serem removidas, o controle de salvamento e o caminho do Parquet."""
        if to_drop is None:
            self.to_drop = [
                'availability_30', 'availability_90', 'calculated_host_listings_count', 'host_total_listings_count',
                'neighbourhood_cleansed', 'maximum_maximum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
                'minimum_minimum_nights', 'property_type', 'number_of_reviews_l30d', 'number_of_reviews_ltm',
                'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'host_response_rate', 'host_neighbourhood'
            ]
        else:
            self.to_drop = to_drop
        
        # Parâmetro que controla se o DataFrame será salvo ou não
        self.salvar = salvar
        self.parquet_path = parquet_path

    def fit(self, X, y=None):
        """Fit não realiza nenhuma operação, necessário para compatibilidade com o Pipeline."""
        return self

    def transform(self, X):
        """Remove as colunas especificadas e salva o DataFrame se o parâmetro salvar for True."""
        # Remover as colunas especificadas
        df_correlacao = X.drop(columns=self.to_drop)
        
        # Salvar o DataFrame em formato Parquet se o parâmetro salvar for True
        if self.salvar:
            self._salvar_parquet(df_correlacao)
        
        return df_correlacao

    def _salvar_parquet(self, df):
        """Salva o DataFrame em formato Parquet no caminho especificado."""
        df.to_parquet(self.parquet_path)

class OneHotImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = None
        self.categorical_cols = None

    def fit(self, X, y=None):
        # Identificar colunas categóricas (antes do get_dummies)
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Criar imputador com a estratégia de mais frequente para colunas categóricas
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.imputer.fit(X[self.categorical_cols])
        
        return self

    def transform(self, X):
        # Fazer uma cópia do DataFrame original
        X_copy = X.copy()

        # Imputar valores nas colunas categóricas
        X_copy[self.categorical_cols] = self.imputer.transform(X_copy[self.categorical_cols])

        # Aplicar one-hot encoding apenas nas colunas categóricas
        X_copy = pd.get_dummies(X_copy, columns=self.categorical_cols, drop_first=True)

        # Selecionar colunas binárias após a transformação com get_dummies
        boolean_cols = X_copy.select_dtypes(include=['bool']).columns
        
        # Converter todas as colunas binárias para 0 e 1
        X_copy[boolean_cols] = X_copy[boolean_cols].astype(int)

        return X_copy

class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, copy=False, salvar=False, parquet_path=None):
        self.n_neighbors = n_neighbors
        self.copy = copy
        self.salvar = salvar
        self.parquet_path = parquet_path

    def fit(self, X, y=None):
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors, copy=self.copy)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        df_knn = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        df_knn.columns = df_knn.columns.str.lower().str.replace(' ', '_')

        if self.salvar and self.parquet_path:
            self._salvar_parquet(df_knn)

        return df_knn

    def _salvar_parquet(self, df):
        df.to_parquet(self.parquet_path)

    def set_parquet_path(self, new_path):
        self.parquet_path = new_path

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
