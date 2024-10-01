from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from rio_airbnb.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, kruskal
import seaborn as sns
from distfit import distfit
from dython import nominal
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
from scipy.stats import kstest, gamma, norm
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
import shap
import math

class AjusteDistribuicoes:
    def __init__(self, coluna='price', alpha=0.05, distr='popular', random_state=75, n_boots=100, n_top=3, n=10_000):
        """Inicializa o objeto DistfitProcessor com os parâmetros do ajuste e bootstrap."""
        self.coluna = coluna
        self.alpha = alpha
        self.distr = distr
        self.random_state = random_state
        self.n_boots = n_boots
        self.n_top = n_top
        self.n = n
        self.dfit = None  # Este será o objeto distfit após o ajuste
    
    def _ajustar(self, df):
        """Ajusta o modelo distfit à coluna especificada do DataFrame."""
        self.dfit = distfit(alpha=self.alpha, distr=self.distr, random_state=self.random_state, verbose=False)
        
        # Ajustar o modelo e realizar o bootstrap
        self.dfit.fit_transform(df[self.coluna], verbose=False)
        self.dfit.bootstrap(
            X=df[self.coluna], 
            n_boots=self.n_boots, 
            alpha=self.alpha, 
            n=self.n, 
            n_top=self.n_top, 
            update_model=True
        )
    
    def _plotar_graficos(self, df):
        """Gera os gráficos de resumo, pdf, cdf e qqplot com base no ajuste do distfit."""
        if self.dfit is None:
            raise ValueError("O modelo distfit ainda não foi ajustado. Chame o método 'ajustar()' primeiro.")
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plotar resumo das distribuições
        self.dfit.plot_summary(ax=axs[0, 0])
        
        # Plotar gráfico de PDF
        self.dfit.plot(chart='pdf', n_top=self.n_top, ax=axs[0, 1])
        
        # Plotar gráfico de CDF
        self.dfit.plot(chart='cdf', n_top=self.n_top, ax=axs[1, 0])
        
        # Plotar gráfico de QQ-Plot
        qq_ax = axs[1, 1]
        self.dfit.qqplot(df[self.coluna].values, n_top=self.n_top, ax=qq_ax)
        
        # Ajustar os limites do gráfico
        price_min = df[self.coluna].min()
        price_max = df[self.coluna].max()
        qq_ax.set_ylim([price_min, price_max])
        qq_ax.set_xlim([price_min, price_max])
        
        # Melhorar o layout e exibir os gráficos
        plt.tight_layout()
        plt.show()
    
    def processar_e_plotar(self, df):
        """Função principal que ajusta o modelo e gera os gráficos."""
        self._ajustar(df)
        self._plotar_graficos(df)

class MatrizCorrelacao:
    def __init__(self, df, var_objetivo):
        """Inicializa a classe com o DataFrame e a variável objetivo, e organiza as colunas."""
        self.df = df
        self.var_objetivo = var_objetivo
        self.features = sorted(c for c in df.columns if c != var_objetivo) + [var_objetivo]

    def _calcular(self):
        """Calcula a matriz de correlação e a máscara do triângulo superior."""
        corr = nominal.associations(
            dataset=self.df[self.features], 
            plot=False, 
            compute_only=True,
            nan_strategy='drop_sample_pairs',
            multiprocessing=True,
            max_cpu_cores=4,
            num_num_assoc='spearman'
        )['corr']

        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr = corr.iloc[1:, :-1] # type: ignore
        mask = mask[1:, :-1]
        return corr, mask

    def _configurar_heatmap(self, corr, mask):
        """Configura e desenha o heatmap da matriz de correlação."""
        cmap = ListedColormap([
            'darkred', 'firebrick', 'red', 'lightcoral', 'white', 
            'lightblue', 'dodgerblue', 'blue', 'darkblue'
        ])
        bounds = [-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(data=corr, mask=mask, cmap=cmap, norm=norm, cbar=False, square=True, 
                    linewidths=0, annot=False, ax=ax)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if not mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='lightgrey', lw=2)) # type: ignore

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.tick_params(left=False, bottom=False)
        return fig, ax

    def _configurar_legenda(self, ax):
        """Adiciona uma legenda personalizada ao gráfico."""
        legend_elements = [
            Patch(facecolor='darkred', edgecolor='darkred', label='< -0.8'),
            Patch(facecolor='firebrick', edgecolor='firebrick', label='-0.8 to -0.6'),
            Patch(facecolor='red', edgecolor='red', label='-0.6 to -0.4'),
            Patch(facecolor='lightcoral', edgecolor='lightcoral', label='-0.4 to -0.2'),
            Patch(facecolor='white', edgecolor='black', label='-0.2 to 0.2'),
            Patch(facecolor='lightblue', edgecolor='lightblue', label='0.2 to 0.4'),
            Patch(facecolor='dodgerblue', edgecolor='dodgerblue', label='0.4 to 0.6'),
            Patch(facecolor='blue', edgecolor='blue', label='0.6 to 0.8'),
            Patch(facecolor='darkblue', edgecolor='darkblue', label='> 0.8')
        ]
        ax.legend(handles=legend_elements, loc='upper right', title='Correlações')

    def gerar(self):
        """Gera a matriz de correlação, configura e exibe o gráfico."""
        corr, mask = self._calcular()
        fig, ax = self._configurar_heatmap(corr, mask)
        self._configurar_legenda(ax)
        fig.text(x=.5, y=.87, s="Matriz de correlação", ha="center", va="top", size=22)
        plt.show()

class ImpactoVariaveisQuantitativas:
    def __init__(self, df, variavel_resposta, variaveis_quantitativas, texto_explicativo=None):
        """Inicializa a classe com o DataFrame, variável resposta, variáveis quantitativas e o texto explicativo."""
        self.df = df
        self.variavel_resposta = variavel_resposta
        self.variaveis_quantitativas = variaveis_quantitativas
        self.texto_explicativo = texto_explicativo

    def _gerar_boxplot(self, ax, var):
        """Cria o boxplot para uma variável quantitativa."""
        quartiles = pd.qcut(self.df[var], 4, duplicates='drop')
        box = sns.boxplot(x=quartiles, y=self.df[self.variavel_resposta], ax=ax, boxprops=dict(facecolor='lightgray'))
        return quartiles, box

    def _alterar_cor_maior_mediana(self, quartiles, box):
        """Altera a cor do boxplot que contém a maior mediana."""
        max_median_idx = self.df.groupby(quartiles)[self.variavel_resposta].median().idxmax()
        box.patches[quartiles.cat.categories.get_loc(max_median_idx)].set_facecolor('steelblue')

    def _teste_hipotese(self, quartiles):
        """Realiza o teste de hipótese entre os quartis."""
        groups = [self.df[self.variavel_resposta][quartiles == cat] for cat in quartiles.cat.categories]
        if len(groups) == 2:
            stat, p_value = mannwhitneyu(groups[0], groups[1])
        else:
            stat, p_value = kruskal(*groups)
        return p_value

    def _ajustar_labels(self, ax, quartiles, var, p_value):
        """Ajusta os rótulos e o título do boxplot."""
        ax.set_xticklabels([f'[{cat.left:.1f}, {cat.right:.1f}]' for cat in quartiles.cat.categories], rotation=0)
        ax.set_ylabel('')
        ax.set_xlabel(f"{var} (p-value: {p_value:.1e})")

    def _configurar_legenda(self):
        """Configura a legenda dos boxplots."""
        legend_handles = [
            Patch(color='steelblue', label='Maior mediana', edgecolor='black', linewidth=1),
            Patch(color='lightgray', label='Outros quartis', edgecolor='black', linewidth=1),    
        ]
        plt.figlegend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False)

    def _adicionar_texto_explicativo(self):
        """Adiciona o texto explicativo abaixo do gráfico, se houver."""
        if self.texto_explicativo:
            plt.figtext(0.02, 0.25, self.texto_explicativo, wrap=True, horizontalalignment='left', fontsize=10)

    def plot(self):
        """Função principal que orquestra a criação dos boxplots e gera o gráfico final."""
        plt.figure(figsize=(11, 11))
        plt.suptitle('Impacto das variáveis quantitativas no preço', fontsize=16, y=1)

        for i, var in enumerate(self.variaveis_quantitativas):
            ax = plt.subplot((len(self.variaveis_quantitativas) // 2) + 1, 2, i + 1)
            quartiles, box = self._gerar_boxplot(ax, var)
            self._alterar_cor_maior_mediana(quartiles, box)
            p_value = self._teste_hipotese(quartiles)
            self._ajustar_labels(ax, quartiles, var, p_value)

        # Configurar legenda e adicionar o texto explicativo (se houver)
        self._configurar_legenda()
        self._adicionar_texto_explicativo()

        # Ajustes finais
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.3)
        plt.show()

class ImpactoVariaveisQualitativas:
    def __init__(self, df, variavel_resposta, variaveis_qualitativas, room_type_mapping=None, response_time_mapping=None, texto_explicativo=None):
        """Inicializa a classe com o DataFrame, variável resposta, variáveis qualitativas e o texto explicativo."""
        self.df = df
        self.variavel_resposta = variavel_resposta
        self.variaveis_qualitativas = variaveis_qualitativas
        self.room_type_mapping = room_type_mapping or {
            'Entire home/apt': 'Home/apt',
            'Private room': 'Private',
            'Others': 'Others'
        }
        self.response_time_mapping = response_time_mapping or {
            'within a day': 'w/i day',
            'within a few hours': 'few hours',
            'a few days or more': 'days',
            'within an hour': 'w/i hour',
        }
        self.texto_explicativo = texto_explicativo

    def _calcular_mediana(self, var):
        """Agrupa os dados pela variável qualitativa e calcula a mediana dos preços."""
        data = self.df.groupby(var)[self.variavel_resposta].median().reset_index()
        return data.sort_values(by=self.variavel_resposta, ascending=False)

    def _realizar_teste_hipotese(self, var, data):
        """Realiza o teste de hipótese entre os grupos."""
        groups = [self.df[self.df[var] == category][self.variavel_resposta] for category in data[var]]
        stat, p_value = mannwhitneyu(groups[0], groups[1]) if len(groups) == 2 else kruskal(*groups)
        return p_value

    def _plotar_barra(self, ax, var, data, p_value):
        """Cria o gráfico de barra com destaque para a maior mediana."""
        sns.barplot(x=data[var], y=data[self.variavel_resposta], ax=ax, color='lightgray')

        # Destacar a barra com maior mediana
        max_height = max([patch.get_height() for patch in ax.patches])
        for patch in ax.patches:
            if patch.get_height() == max_height:
                patch.set_facecolor('steelblue')
            else:
                patch.set_facecolor('lightgray')
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)

        # Ajustar rótulos e limites
        ax.set_ylim(0, 450)
        ax.set_xlabel(f"{var} (p-value: {p_value:.3e})")
        ax.set_ylabel('')

    def _ajustar_rótulos(self, ax, var, data):
        """Ajusta os rótulos das categorias para melhor visualização."""
        if var == 'room_type':
            ax.set_xticklabels([self.room_type_mapping.get(label, label) for label in data[var]])
        if var == 'host_response_time':
            ax.set_xticklabels([self.response_time_mapping.get(label, label) for label in data[var]])

    def _configurar_legenda(self):
        """Configura a legenda dos gráficos de barra."""
        legend_handles = [
            Patch(color='steelblue', label='Maior mediana', edgecolor='black', linewidth=1),
            Patch(color='lightgray', label='Outros quartis', edgecolor='black', linewidth=1),
        ]
        plt.figlegend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False)

    def _adicionar_texto_explicativo(self):
        """Adiciona o texto explicativo abaixo do gráfico, se houver."""
        if self.texto_explicativo:
            plt.figtext(0.02, 0.18, self.texto_explicativo, wrap=True, horizontalalignment='left', fontsize=10)

    def plot(self):
        """Função principal que orquestra a criação dos gráficos de barra e gera o gráfico final."""
        plt.figure(figsize=(12, 12))
        plt.suptitle('Impacto das variáveis qualitativas no preço', fontsize=16, y=1)
        num_vars = len(self.variaveis_qualitativas)

        for i, var in enumerate(self.variaveis_qualitativas):
            # Calcular a mediana e o teste de hipótese
            data = self._calcular_mediana(var)
            p_value = self._realizar_teste_hipotese(var, data)

            # Verificar se a diferença é significativa
            if p_value < 0.05:
                ax = plt.subplot((num_vars // 3) + 1, 3, i + 1)
                self._plotar_barra(ax, var, data, p_value)
                self._ajustar_rótulos(ax, var, data)

        # Configurar legenda e adicionar o texto explicativo (se houver)
        self._configurar_legenda()
        self._adicionar_texto_explicativo()

        # Ajustes finais
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.3)
        plt.show()

class MapaMedianaPrecos:
    def __init__(self, df, tamanho_grade=0.01):
        """Inicializa a classe com o DataFrame e o tamanho da grade."""
        self.df = df.copy()
        self.tamanho_grade = tamanho_grade
        self.medianas_precos = None
        self.centro_mapa = [df['latitude'].mean(), df['longitude'].mean()]
    
    def _calcular_mediana_precos(self):
        """Calcula a mediana dos preços para cada célula da grade definida por tamanho_grade."""
        # Criar colunas para a grade
        self.df['grade_lat'] = (self.df['latitude'] // self.tamanho_grade) * self.tamanho_grade
        self.df['grade_lon'] = (self.df['longitude'] // self.tamanho_grade) * self.tamanho_grade
        
        # Agrupar por células da grade e calcular a mediana dos preços
        self.medianas_precos = self.df.groupby(['grade_lat', 'grade_lon'])['price'].median().reset_index()

        # Retornar uma lista de [latitude, longitude, mediana]
        dados_calor = self.medianas_precos[['grade_lat', 'grade_lon', 'price']].values.tolist()
        return dados_calor
    
    def _criar_colormap(self):
        """Cria o colormap para os preços."""
        min_price = self.df['price'].min()
        max_price = self.df['price'].max()
        colormap = cm.LinearColormap(['blue', 'lime', 'red'], vmin=min_price, vmax=max_price)
        colormap.caption = 'Escala de Preços'
        return colormap
    
    def _adicionar_legenda_customizada(self, mapa, colormap):
        """Adiciona uma legenda personalizada ao mapa."""
        colormap_html = colormap._repr_html_().replace(colormap.caption, '')
        legend_html = f"""
        <div style="
        position: fixed; 
        bottom: 20px; 
        right: 20px; 
        z-index:9999; 
        background-color: rgba(255, 255, 255, 0.7); 
        padding: 10px; 
        border-radius: 5px; 
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.2);
        ">
        <h4 style="margin:0;">Mediana dos preços a cada 100 metros</h4>
        {colormap_html}
        </div>
        """
        mapa.get_root().html.add_child(folium.Element(legend_html))
    
    def gerar(self, zoom_start=11, radius=10, blur=10, min_opacity=0.3):
        """Gera o mapa de calor interativo com base nas medianas dos preços."""
        # Calcular os dados do HeatMap
        heatmap = self._calcular_mediana_precos()
        
        # Criar o mapa centrado
        mapa = folium.Map(location=self.centro_mapa, zoom_start=zoom_start)
        
        # Adicionar o HeatMap ao mapa
        HeatMap(heatmap, radius=radius, blur=blur, min_opacity=min_opacity).add_to(mapa)
        
        # Criar e adicionar o colormap personalizado
        colormap = self._criar_colormap()
        self._adicionar_legenda_customizada(mapa, colormap)
        
        return mapa

class PremissasGama:
    def __init__(self, modelo):
        """
        Inicializa a classe com o modelo fornecido.
        """
        self.modelo = modelo
        self.residuals = modelo.resid_deviance

    def distribuicao_residuos_gama(self):
        """
        Ajusta a distribuição gama aos resíduos do modelo e plota os gráficos.
        Retorna a estatística KS e o valor p.
        """
        dfit = distfit(distr='gamma')
        result = dfit.fit_transform(self.residuals, verbose=False)
        dfit.bootstrap(X=self.residuals, n_boots=100, alpha=0.05, n=10_000, update_model=True)
        
        ks_stat, p_value = kstest(self.residuals, 'gamma', args=result['model']['params'])
        
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle("Distribuição dos resíduos", fontsize=16, y=1)
        
        dfit.plot(chart='pdf', ax=axs[0])
        dfit.plot(chart='cdf', ax=axs[1])
        axs[0].set_title('Função Densidade de Probabilidade (PDF)', fontsize=12)
        axs[1].set_title('Função Distribuição Cumulativa (CDF)', fontsize=12)
        
        for ax in axs:
            ax.set_xlabel('Valores', fontsize=10)
            ax.set_ylabel('Frequência', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.figtext(0.02, -0.05, 
                    (f"Distribuição Ajustada: "
                     f"gamma(a={result['model']['params'][0]:.4f}, "
                     f"loc={result['model']['params'][1]:.4f}, "
                     f"scale={result['model']['params'][2]:.6f}) \n"
                     f"Teste de Kolmogorov-Smirnov: "
                     f"KS statistic: {ks_stat:.2f}, p-value: {p_value:.2e}"),
                    wrap=True, horizontalalignment='left', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()

        return ks_stat, p_value

    def envel_gama(self):
        """
        Baseado na função envel_gama em R do professor Gilberto A. Paula
        """
        X = self.modelo.model.exog
        n = X.shape[0]
        p = X.shape[1]
        ro = self.modelo.resid_response
        h = self.modelo.get_influence().hat_matrix_diag
        fi = (n - p) / np.sum((ro / self.modelo.fittedvalues) ** 2)
        td = self.modelo.resid_deviance * np.sqrt(fi / (1 - h))
        e = np.zeros((n, 100))

        for i in range(100):
            resp = gamma.rvs(a=fi, size=n)
            resp = (self.modelo.fittedvalues / fi) * resp
            fit = sm.GLM(resp, X, family=Gamma(link=sm.families.links.Log())).fit()
            ro = fit.resid_response
            h = fit.get_influence().hat_matrix_diag
            phi = (n - p) / np.sum((ro / fit.fittedvalues) ** 2)
            e[:, i] = np.sort(fit.resid_deviance * np.sqrt(phi / (1 - h)))
        
        e1 = np.percentile(e, 2.5, axis=1)
        e2 = np.percentile(e, 97.5, axis=1)
        med = np.mean(e, axis=1)
        faixa = (min(td.min(), e1.min(), e2.min()), max(td.max(), e1.max(), e2.max()))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        norm_qq = norm.ppf((np.arange(1, n+1) - 0.5) / n)
        ax.scatter(norm_qq, np.sort(td), color='black', s=2, label='Resíduos')
        ax.plot(norm_qq, np.sort(e1), color='red', linestyle='--', label='Envelope Inferior (2.5%)')
        ax.plot(norm_qq, np.sort(e2), color='red', linestyle='--', label='Envelope Superior (97.5%)')
        ax.plot(norm_qq, np.sort(med), color='blue', linestyle='-', label='Média Simulada')

        ax.set_xlabel("Percentil da N(0,1)", fontsize=10)
        ax.set_ylabel("Componente do Desvio", fontsize=10)
        ax.set_ylim(faixa)
        ax.set_xlim(faixa)
        ax.legend(loc='best', fontsize=8)
        ax.set_title("Envelope dos resíduos", fontsize=12)
        fig.show()

    def gamma_shape(self):
        """
        Calcula o parâmetro de forma da distribuição Gama.
        """
        mean_deviance = np.mean(self.modelo.resid_deviance**2)
        phi = mean_deviance / (self.modelo.nobs - self.modelo.df_model - 1)
        return phi

    def diag_gama(self):
        """
        Baseado na função diag_gama em R do professor Gilberto A. Paula
        """
        X = self.modelo.model.exog
        n = X.shape[0]
        p = X.shape[1]
        w = self.modelo.model.weights
        h = self.modelo.get_influence().hat_matrix_diag
        fi = self.gamma_shape()
        ts = self.modelo.resid_pearson * np.sqrt(fi / (1 - h))
        td = self.modelo.resid_deviance * np.sqrt(fi / (1 - h))
        di = (h / (1 - h)) * (ts ** 2)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        scatter_params = {'c': 'black', 'marker': 'o', 's': 2}

        axs[0, 0].scatter(self.modelo.fittedvalues, h, **scatter_params)
        axs[0, 0].axhline(2 * p / n, color='red', linestyle='dashed')
        axs[0, 0].set_xlabel("Valor Ajustado")
        axs[0, 0].set_ylabel("Medida h")
        axs[0, 0].set_title("Medida h vs Valor Ajustado")

        axs[0, 1].scatter(np.arange(n), di, **scatter_params)
        axs[0, 1].axhline(4 / n, color='red', linestyle='dashed')
        axs[0, 1].set_xlabel("Índice")
        axs[0, 1].set_ylabel("Distância de Cook")

        a = np.max(td)
        b = np.min(td)
        axs[1, 0].scatter(self.modelo.fittedvalues, td, **scatter_params)
        axs[1, 0].set_xlabel("Valor Ajustado")
        axs[1, 0].set_ylabel("Resíduo Componente do Desvio")
        axs[1, 0].set_ylim([min(b - 1, -2.5), max(a + 1, 2.5)])
        axs[1, 0].axhline(2, color='red', linestyle='dashed')
        axs[1, 0].axhline(-2, color='red', linestyle='dashed')

        eta = self.modelo.predict()
        z = eta + self.modelo.resid_pearson / np.sqrt(w)
        axs[1, 1].scatter(eta, z, **scatter_params)
        axs[1, 1].plot([min(eta), max(eta)], [min(eta), max(eta)], color='red', linestyle='dashed')
        axs[1, 1].set_xlabel("Preditor Linear")
        axs[1, 1].set_ylabel("Variável z")

        plt.tight_layout()
        fig.show()

class InterpretaCoeficientes:
    def __init__(self, summary):
        # Extrair a tabela de coeficientes do resumo
        coef_table = summary.tables[1].data

        # Converter a tabela para um DataFrame
        self.coef_df = pd.DataFrame(coef_table[1:], columns=coef_table[0])

        # Renomear colunas para facilitar o uso
        self.coef_df.columns = ['variavel', 'coeficiente', 'std_err', 'z', 'p>|z|', '[0.025', '0.975]']

        # Converter a coluna de coeficiente para tipo numérico
        self.coef_df['coeficiente'] = self.coef_df['coeficiente'].astype(float)

        # Calcular a transformação \(e^{\text{coeficiente}} - 1\)
        self.coef_df['transformacao'] = (np.exp(self.coef_df['coeficiente']) - 1) * 100
        self.coef_df = self.coef_df.sort_values(by='transformacao', ascending=True)[self.coef_df['variavel'] != 'Intercept']
        
        self.interpretacoes = {
            'longitude': '{Aumento/Redução} de {Porcentagem}% no preço para cada unidade de aumento na longitude.',
            'latitude': '{Aumento/Redução} de {Porcentagem}% no preço para cada unidade de aumento na latitude.',
            'room_type_others': '{Aumento/Redução} de {Porcentagem}% no preço para outros tipos de quartos.',
            'host_responded': '{Aumento/Redução} de {Porcentagem}% no preço se o anfitrião respondeu.',
            'room_type_private_room': '{Aumento/Redução} de {Porcentagem}% no preço para quartos privados.',
            'was_reviewed': '{Aumento/Redução} de {Porcentagem}% no preço se o imóvel foi avaliado.',
            'review_scores_location': '{Aumento/Redução} de {Porcentagem}% no preço para cada ponto adicional na pontuação de localização.',
            'review_scores_value': '{Aumento/Redução} de {Porcentagem}% no preço para cada ponto adicional na pontuação de valor.',
            'bedrooms': '{Aumento/Redução} de {Porcentagem}% no preço para cada quarto adicional.',
            'review_scores_rating': '{Aumento/Redução} de {Porcentagem}% no preço para cada ponto adicional na pontuação geral.',
            'bathrooms': '{Aumento/Redução} de {Porcentagem}% no preço para cada banheiro adicional.',
            'reviews_per_month': '{Aumento/Redução} de {Porcentagem}% no preço para cada avaliação por mês adicional.',
            'review_scores_cleanliness': '{Aumento/Redução} de {Porcentagem}% no preço para cada ponto adicional na pontuação de limpeza.',
            'review_scores_communication': '{Aumento/Redução} de {Porcentagem}% no preço para cada ponto adicional na pontuação de comunicação.',
            'review_scores_accuracy': '{Aumento/Redução} de {Porcentagem}% no preço para cada ponto adicional na pontuação de precisão.',
            'verification_work_email': '{Aumento/Redução} de {Porcentagem}% no preço se o anfitrião usa e-mail de trabalho verificado.',
            'host_response_time_within_a_day': '{Aumento/Redução} de {Porcentagem}% no preço se o anfitrião responde dentro de um dia.',
            'accommodates': '{Aumento/Redução} de {Porcentagem}% no preço para cada pessoa adicional acomodada.',
            'instant_bookable': '{Aumento/Redução} de {Porcentagem}% no preço para imóveis com reserva instantânea.',
            'host_response_time_within_an_hour': '{Aumento/Redução} de {Porcentagem}% no preço se o anfitrião responde em até uma hora.',
            'host_acceptance_rate': '{Aumento/Redução} de {Porcentagem}% no preço para cada aumento percentual na taxa de aceitação do anfitrião.',
            'host_is_superhost': '{Aumento/Redução} de {Porcentagem}% no preço se o anfitrião é Superhost.',
            'calculated_host_listings_count_shared_rooms': '{Aumento/Redução} de {Porcentagem}% no preço para cada listagem de quarto compartilhado do anfitrião.',
            'beds': '{Aumento/Redução} de {Porcentagem}% no preço para cada cama adicional.',
            'verification_email': '{Aumento/Redução} de {Porcentagem}% no preço se o anfitrião usa e-mail verificado.',
            'num_amenities': '{Aumento/Redução} de {Porcentagem}% no preço para cada comodidade adicional.',
            'minimum_nights': '{Aumento/Redução} de {Porcentagem}% no preço para cada noite mínima adicional requerida.',
            'availability_60': '{Aumento/Redução} de {Porcentagem}% no preço para cada dia adicional de disponibilidade nos próximos 60 dias.',
            'availability_365': '{Aumento/Redução} de {Porcentagem}% no preço para cada dia adicional de disponibilidade.',
            'host_listings_count': '{Aumento/Redução} de {Porcentagem}% no preço para cada listagem adicional do anfitrião.',
            'days_since_last_review': '{Aumento/Redução} de {Porcentagem}% no preço para cada dia adicional desde a última avaliação.',
            'number_of_reviews': '{Aumento/Redução} de {Porcentagem}% no preço para cada avaliação adicional.',
            'host_about_length': '{Aumento/Redução} de {Porcentagem}% no preço para cada caractere adicional na descrição do anfitrião.',
            'days_since_first_review': '{Aumento/Redução} de {Porcentagem}% no preço para cada dia adicional desde a primeira avaliação.',
            'maximum_nights': '{Aumento/Redução} de {Porcentagem}% no preço para cada noite máxima adicional permitida.',
            'days_since_host_active': '{Aumento/Redução} de {Porcentagem}% no preço para cada dia adicional desde que o anfitrião foi ativo.'
        }

    def gerar_interpretacao(self, variavel, valor_transformacao):
        # Verifica se a variável tem uma interpretação pré-definida
        base_texto = self.interpretacoes.get(variavel, None)
        
        # Se não houver interpretação definida para a variável, retornar uma mensagem padrão ou ignorar
        if base_texto is None:
            return f'Interpretação não disponível para a variável {variavel}'
        
        # Determinar se o valor é aumento ou redução
        if valor_transformacao > 0:
            direcao = "Aumento"
        else:
            direcao = "Redução"
        
        # Inserir a porcentagem absoluta no texto
        porcentagem = abs(valor_transformacao)
        interpretacao = base_texto.replace('{Aumento/Redução}', direcao).replace('{Porcentagem}', f'{porcentagem:.2f}')
        return interpretacao

    def plotar_coeficientes(self):
        plt.figure(figsize=(12, 10))
    
        # Plotando as barras horizontalmente
        plt.hlines(y=self.coef_df['variavel'], xmin=0, xmax=self.coef_df['transformacao'], color='dimgrey', linewidth=2)
    
        # Adicionando as bolas com as cores especificadas e labels
        for i in range(len(self.coef_df)):
            valor = self.coef_df.iloc[i]['transformacao']
            label = self.coef_df.iloc[i]['variavel']
    
            # Verifica se existe uma interpretação para a variável
            if label not in self.interpretacoes:
                continue  # Pula a variável se não houver interpretação
    
            interpretacao = self.gerar_interpretacao(label, valor)

            # Determinar a cor da bola e o contorno, contorno darkgrey para todas
            if -1 <= valor <= 1:
                plt.plot(valor, label, 'o', markerfacecolor='white', markersize=8, markeredgewidth=2, markeredgecolor='dimgrey', zorder=3)
                number_color = 'dimgrey'
            elif valor > 1:
                plt.plot(valor, label, 'o', markerfacecolor='green', markersize=8, markeredgewidth=2, markeredgecolor='dimgrey', zorder=3)
                number_color = 'green'
            else:
                plt.plot(valor, label, 'o', markerfacecolor='red', markersize=8, markeredgewidth=2, markeredgecolor='dimgrey', zorder=3)
                number_color = 'red'

            # Ajustando a posição do label para o lado oposto ao valor com equidistância
            if valor > 0:
                plt.text(valor + 3, label, f"{valor:.2f}", va='center', ha='left', color=number_color, fontweight='bold')
                plt.text(valor + 20, label, label, va='center', ha='left', color='black')
                plt.text(-5, label, interpretacao, va='center', ha='right', color='dimgrey', fontsize=8)
            else:
                plt.text(valor - 20, label, label, va='center', ha='right', color='black')
                plt.text(valor - 3, label, f"{valor:.2f}", va='center', ha='right', color=number_color, fontweight='bold')
                plt.text(5, label, interpretacao, va='center', ha='left', color='dimgrey', fontsize=8)

        # Adicionando a linha vertical central percorrendo todo o gráfico
        plt.axvline(x=0, color='dimgrey', linestyle='-', linewidth=2, zorder=2, ymin=0, ymax=1)

        # Adicionando o título
        plt.title('Impacto das Variáveis no Preço: Análise de Coeficientes e Interpretações')

        # Removendo ticks dos eixos x e y
        plt.xticks([])
        plt.yticks([])

        # Removendo grid e contorno
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.grid(False)

        # Exibindo o gráfico
        plt.show()

class ShapPlotter:

    def __init__(self, model, X_train, X_test):
        """Inicializa a classe com o modelo, dados de treino e teste."""
        shap.initjs()
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = shap.Explainer(self.model, self.X_train)
        self.shap_values = self.explainer(self.X_test)

    def dependence_plot(self, feature_names, max_columns=2, interaction_index=None):
        """Gera gráficos de dependência parcial para as features especificadas."""
        shap_values_array = np.array(self.shap_values.values)  # Extração dos valores SHAP corretos
        
        num_features = len(feature_names)
        num_rows = (num_features + max_columns - 1) // max_columns  # Calcula o número de linhas dinamicamente

        # Criar a grade de subplots
        fig, axs = plt.subplots(num_rows, max_columns, figsize=(max_columns * 6, num_rows * 5))
        
        # Achatar axs se houver várias linhas
        axs = axs.flatten() if num_rows > 1 else [axs]

        # Iterar sobre as features e plotar os gráficos de dependência parcial manualmente em cada posição
        for i, feature in enumerate(feature_names):
            plt.sca(axs[i])  # Define o subplot atual como o ativo
            shap.dependence_plot(feature, shap_values_array, self.X_test, interaction_index=interaction_index, show=False)  # Evita mostrar os gráficos imediatamente

        # Ajustar o layout dos subplots
        plt.tight_layout()
        plt.show()

def plot_feature_importance(model, feature_names, max_num_features=None):
    """Plota o gráfico de importância das features para o modelo fornecido, com barras horizontais."""
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("O modelo fornecido não suporta 'feature_importances_'")

    # Obter a importância das features
    importance = model.feature_importances_
    
    # Ordenar as importâncias em ordem decrescente
    indices = np.argsort(importance)[::-1]
    
    # Se max_num_features estiver definido, limitar o número de features
    if max_num_features is not None:
        indices = indices[:max_num_features]
    
    # Plotar o gráfico com barras horizontais
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(len(indices)), importance[indices], align="center")
    plt.yticks(range(len(indices)), np.array(feature_names)[indices]) # type: ignore
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Inverter para ter as features mais importantes no topo
    plt.show()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
