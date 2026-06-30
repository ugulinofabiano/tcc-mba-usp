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

- ✅ **Janela temporal dupla** — observação e predição estritamente separadas (12 meses de histórico + 6 meses de previsão)
- ✅ **Validação empírica** — demonstra quantitativamente o filtro de compradores únicos
- ✅ **Protocolo multi-métrica** — avalia AUC, F1-Macro, MCC, G-Mean, Kappa e Precision-Recall em conjunto
- ✅ **Representatividade do balanceamento** — etapa rara em estudos aplicados, mas essencial para validade interna

---

## 📁 Estrutura do Repositório

```
tcc-mba-usp/
├── notebook/
│   └── olist_churn_prediction.ipynb    # Notebook principal com análise completa
├── data/
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   └── product_category_name_translation.csv
├── requirements.txt                     # Dependências do projeto
├── guia_notebook.md                     # Documentação técnica detalhada
└── README.md                            # Este arquivo
```

---

## 🛠️ Tecnologias e Dependências

### Bibliotecas Principais

```
Python 3.8+
pandas           # Manipulação e transformação de dados
numpy            # Computação numérica
scikit-learn     # Modelos de ML, métricas e pré-processamento
matplotlib       # Visualização de gráficos
seaborn          # Visualização estatística avançada
xgboost          # Gradient boosting
imbalanced-learn # Tratamento de desbalanceamento
pycaret          # AutoML e experimentação rápida
shap             # Explicabilidade de modelos (SHAP values)
scipy            # Testes estatísticos (PSI, KS-test)
joblib           # Serialização de modelos
```

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
- **Churn**: Variável alvo (não comprou nos 6 meses seguintes)

---

## 📈 Etapas da Análise

O notebook é estruturado em **11 seções principais**:

0. **Imports e Configurações** — Carregamento de bibliotecas e parâmetros globais
1. **Carregamento de Dados** — Leitura dos CSVs relacionais
2. **Exploração Inicial** — Estatísticas descritivas e qualidade dos dados
3. **Limpeza e Tratamento** — Tratamento de valores ausentes, duplicatas, outliers
4. **Feature Engineering** — Criação de variáveis comportamentais
5. **Análise Descritiva** — Características dos churners vs. retentores
6. **Balanceamento de Classes** — Técnicas para dataset desbalanceado
7. **Pré-Processamento** — Normalização, encoding, seleção de features
8. **Modelagem** — Treinamento de 10+ algoritmos
9. **Validação** — Avaliação com métricas múltiplas
10. **Explicabilidade** — SHAP values e interpretação de features

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

**Última atualização**: 2024
