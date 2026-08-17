# Análise Preditiva de Churn em E-Commerce Brasileiro

**Projeto de TCC — MBA em Inteligência Artificial e Big Data | ICMC/USP**

---

## 📋 Sobre o Projeto

Este repositório contém uma análise preditiva abrangente sobre **identificação de clientes com risco de churn** na plataforma Olist, o maior marketplace de e-commerce brasileiro. 

O trabalho constrói um modelo de machine learning capaz de antecipar, com base em dados transacionais históricos, quais clientes têm maior propensão a não voltar a comprar. A abordagem enfatiza rigor metodológico, validação robusta e eliminação completa de vazamento de informação (*data leakage*).

---

## 🎯 Problema e Solução

### O Problema
A Olist conecta pequenos varejistas a grandes canais de venda. Como em qualquer e-commerce, uma parcela significativa de clientes realiza uma única compra e nunca retorna — o **churn**. 

Antecipar esse comportamento permite que a empresa direcione esforços de retenção (cupons, ofertas, contato) de forma proativa, maximizando ROI e reduzindo custos de aquisição.

### A Pergunta de Pesquisa
> Como identificar, de forma antecipada e com base apenas em dados transacionais históricos, quais clientes da Olist apresentam maior propensão a não voltar a comprar?

### Metodologia Diferenciada
Este pipeline se destaca por:

- ✅ **Janela temporal dupla** — observação (set/2016 a dez/2017, ~16 meses) e predição (jan a mai/2018, 5 meses) estritamente separadas
- ✅ **Validação empírica** — demonstra quantitativamente, por ablação, que excluir compradores únicos melhora o modelo (não apenas argumento teórico)
- ✅ **Protocolo multi-métrica** — avalia AUC, F1-Macro, MCC, G-Mean, Kappa e Average Precision em conjunto, com análise de correlação de Spearman entre rankings
- ✅ **Representatividade do balanceamento** — validação por PSI e KS-test, etapa rara em estudos aplicados, mas essencial para validade interna
- ✅ **Estabilidade da importância de features** — checagem em 30 partições estratificadas e comparação com importância por permutação, em vez de tratar a concordância Gini/SHAP como validação independente

---

## 📁 Estrutura do Repositório

```
tcc-mba-usp/
├── notebook/
│   ├── olist_churn_prediction.ipynb           # Notebook principal com análise completa
│   ├── olist_churn_predictions.csv            # Predições geradas pelo modelo final (1.178 clientes)
│   ├── docs/                                  # Versões do texto do TCC e análises de apoio
│   │   ├── tcc_churn_v6.1.md / .docx / .pdf
│   │   ├── analise_distribuicao_score_churn.md
│   │   └── analise_waterfall_shap.md
│   └── imgs/                                  # Imagens de apoio (diagrama entidade-relacionamento etc.)
├── data/
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
├── requirements.txt                     # Dependências do projeto
├── guia_notebook.md                     # Documentação técnica detalhada
└── README.md                            # Este arquivo
```

> As figuras citadas no TCC (`figura1_pipeline.png`, `figura2_janelas.png`, `figura9_scores.png`) não ficam versionadas prontas: são geradas na pasta `notebook/` ao executar as células correspondentes (Seções 9 e final do notebook).

---

## 🛠️ Tecnologias e Dependências

### Bibliotecas Principais

Efetivamente importadas e usadas pelo notebook:

```
Python 3.8+
pandas           # Manipulação e transformação de dados
numpy            # Computação numérica
scikit-learn     # Os 10 algoritmos do benchmark, métricas e pré-processamento
imbalanced-learn # SMOTE, usado na comparação de estratégias de balanceamento (Seção 6)
matplotlib       # Visualização de gráficos
seaborn          # Visualização estatística avançada
shap             # Explicabilidade de modelos (SHAP values, Seção 8.6)
scipy            # Testes estatísticos (correlação de Spearman, PSI, KS-test)
```

`requirements.txt` também instala `pycaret`, `xgboost`, `joblib`, `ipython` e `ipywidgets` para dar suporte ao ambiente Jupyter e a experimentos futuros, mas `pycaret` e `xgboost` não são usados na versão atual do notebook.

### Instalação

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Usar

### Executar o Notebook

1. **Clone ou extraia o repositório**
   ```bash
   cd tcc-mba-usp
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Inicie o Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Abra e execute o notebook**
   - Navegue até `notebook/olist_churn_prediction.ipynb`
   - Execute as células na sequência (Shift + Enter)

### Reprodutibilidade
O notebook foi desenvolvido com foco em reprodutibilidade:
- Seed global fixado em `42` para garantir determinismo
- Todas as transformações estão documentadas
- Parâmetros-chave centralizados no início do notebook

---

## 📊 Dataset

### Origem
**Olist Brazilian E-Commerce Public Dataset** (Kaggle)
- ~100 mil pedidos reais
- Período: 2016–2018
- Maior base pública de e-commerce brasileiro

### Tabelas Relacionais

| Tabela | Descrição |
|--------|-----------|
| `orders` | Pedidos com timestamps e status |
| `customers` | Dados de clientes (localização, etc.) |
| `order_items` | Itens de cada pedido |
| `order_payments` | Formas e parcelamento de pagamento |
| `order_reviews` | Avaliações de clientes (satisfação) |
| `products` | Catálogo com categorias de produtos |
| `sellers` | Vendedores terceirizados |
| `geolocation` | Dados geográficos (CEP) |

### Variáveis Principais
- **Recência**: Dias desde última compra
- **Frequência**: Número total de compras
- **Valor Monetário**: Gasto total e ticket médio
- **Satisfação**: Média de avaliações
- **Padrão de Pagamento**: Modalidade, parcelamento
- **Engajamento**: Diversidade de categorias
- **Churn**: Variável alvo (não comprou na janela de predição de 5 meses seguintes, jan a mai/2018)

---

## 📈 Etapas da Análise

O notebook segue **12 seções principais** (algumas com subseções de validação empírica):

| Seção | Conteúdo |
|-------|----------|
| 0 | Imports e configurações globais (janelas temporais, paleta de cores, seed) |
| 1 | Carregamento das seis tabelas relacionais da Olist |
| 1.1 | Consultas exploratórias: volumetria, recorrência de compra e diversidade de categorias |
| 2 | Limpeza, conversão de datas e split temporal (observação × predição) |
| 3 | Feature engineering: as onze variáveis RFM + comportamentais |
| 4 | Definição do target (`churn`) e filtro `frequency ≥ 2` |
| 4.1 | Validação empírica da exclusão de compradores únicos (ablação) |
| 5 | Análise exploratória dos dados (EDA) — Ativos vs. Churn |
| 6 | Pré-processamento, balanceamento e comparação empírica de estratégias (30 splits) |
| 7 | Benchmark dos dez algoritmos, curvas ROC/PR e ponto de corte ótimo |
| 7.1 | Robustez do ranking: correlação de Spearman entre AUC, MCC, G-Mean e Kappa |
| 8 | Interpretabilidade via importância de Gini (Random Forest) |
| 8.5 | Validação de representatividade do undersampling (PSI e KS-test) |
| 8.6 | Análise SHAP (direção dos efeitos, dependence e waterfall plots) |
| — | Estabilidade da importância entre 30 partições e comparação com importância por permutação |
| 9 | Score de churn, segmentação de risco em quatro faixas e diagnósticos de validação |
| 10 | Exportação do CSV de predições |

Para o passo a passo detalhado célula a célula, com números da execução mais recente, ver **[guia_notebook.md](guia_notebook.md)**.

---

## 🏆 Principais Resultados

Números da execução de referência (seed 42), sobre a base filtrada de **1.178 clientes** (1.128 Churn, 50 Ativos):

| Indicador | Valor |
|-----------|-------|
| Modelo selecionado | Random Forest |
| ROC-AUC (hold-out) | 0,8106 |
| Recall Churn | 97,4% |
| MCC / Kappa (hold-out) | 0,1931 / 0,1918 |
| Limiar de decisão adotado | 0,38 (derivado do treino, maximiza Kappa) |
| Features mais importantes (Gini) | `frequency` (23,2%), `recency_days` (12,2%), `n_categories` (11,9%) |
| Representatividade do undersampling | 8 de 9 features com PSI < 0,10; nenhuma distorcida |
| Clientes segmentados | 1.178, em 4 faixas de risco (🟢 Baixo 2,3% · 🟡 Médio 8,6% · 🟠 Alto 31,0% · 🔴 Crítico 58,1%) |

O Random Forest não é o melhor modelo em todas as métricas: a Regressão Logística lidera em MCC e Kappa, e o HistGradient Boosting em G-Mean. A escolha por AUC é justificada, mas não confirmada de forma unânime pelas métricas robustas (Seção 7.1); detalhes em [guia_notebook.md](guia_notebook.md).

---

## 📖 Documentação Adicional

Para uma compreensão completa e sequencial do pipeline:

👉 **[guia_notebook.md](guia_notebook.md)** — Documentação técnica detalhada

Este guia explica, para cada seção:
- **O que faz**: Objetivo da etapa
- **Por que faz assim**: Justificativa metodológica
- **Como faz**: Descrição técnica e código relevante
- **Resultado obtido**: Métricas e números reais da execução

---

## 👨‍💼 Autor

**Fabiano Ugulino**  
Mestrando em IA e Big Data | ICMC/USP

**Orientadora**  
Profa. Dra. Cibele M. Russo

---

## 📝 Licença

Este projeto é fornecido para fins educacionais e de pesquisa.

---

## 🔗 Referências

- **Dataset**: [Olist Brazilian E-Commerce Public Dataset - Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- **Instituição**: [ICMC - Instituto de Ciências Matemáticas e de Computação (USP)](https://www.icmc.usp.br/)
- **Programa**: MBA em Inteligência Artificial e Big Data

---

**Última atualização**: 2026
