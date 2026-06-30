# Guia Completo — Notebook `Olist_Churn_Prediction_v12`

**Análise Preditiva de Churn em E-Commerce Brasileiro**
TCC — MBA em Inteligência Artificial e Big Data | ICMC/USP
Autor: Fabiano Ugulino | Orientadora: Profa. Dra. Cibele M. Russo

---

## Sobre este guia

Este documento explica, em ordem sequencial, **o que o notebook faz, por que faz e o que ele encontra**. O objetivo é que qualquer pessoa — mesmo sem familiaridade com Machine Learning aplicado a dados de e-commerce — consiga acompanhar o raciocínio do início ao fim, entender as decisões metodológicas e reproduzir o trabalho.

A estrutura segue exatamente a do notebook, organizada em onze etapas (Seções 0 a 10, com subseções 4.1, 7.1, 7.2, 7.3, 8.5 e 8.6). Cada etapa é apresentada em quatro blocos: **O que faz**, **Por que faz assim** (justificativa metodológica), **Como faz** (descrição técnica) e **Resultado obtido** (números reais da execução).

---

## Visão geral da proposta

### O problema

A Olist é uma plataforma de marketplace que conecta pequenos varejistas brasileiros a grandes canais de venda (Mercado Livre, Americanas, etc.). Como em todo e-commerce, uma parcela significativa dos clientes compra uma única vez e nunca mais retorna — fenômeno conhecido como **churn**. Antecipar quais clientes têm maior risco de abandono permite que a empresa direcione esforços de retenção (cupons, ofertas, contato) de forma proativa, em vez de reagir à perda já consumada.

### A pergunta de pesquisa

> Como identificar, de forma antecipada e com base apenas em dados transacionais históricos, quais clientes da Olist apresentam maior propensão a não voltar a comprar — e qual é o perfil do cliente que retorna?

### A abordagem em uma frase

Construir um modelo preditivo que, a partir de variáveis comportamentais de cada cliente (recência, frequência, valor, satisfação, padrão de pagamento, diversidade de categorias), estime a probabilidade de churn nos próximos seis meses, com rigor metodológico para evitar vazamento de informação e validar a generalização.

### O que torna este trabalho diferente

Quatro elementos distinguem este pipeline da maioria dos estudos comparáveis sobre o dataset Olist:

1. **Janela temporal dupla** — observação e predição estritamente separadas, eliminando data leakage
2. **Validação empírica do filtro de compradores únicos** — não é argumento puramente teórico, é demonstrado quantitativamente
3. **Protocolo multi-métrica** — avalia AUC, F1-Macro, MCC, G-Mean, Kappa e Precision-Recall em conjunto, respondendo a uma limitação metodológica documentada na literatura recente
4. **Validação de representatividade do balanceamento** — etapa raramente conduzida em estudos aplicados, mas essencial para defender a validade interna dos resultados

### As bibliotecas

```python
pandas, numpy           # manipulação e estatística
scikit-learn            # modelos, métricas, pré-processamento
matplotlib, seaborn     # visualização
shap                    # explicabilidade
scipy                   # PSI e KS-test
```

---

## Seção 0 — Imports e Configurações Globais

### O que faz

Carrega todas as bibliotecas necessárias e define três conjuntos de constantes globais que governam o restante do pipeline: as janelas temporais, a paleta de cores e o filtro de warnings.

### Por que faz assim

**Reprodutibilidade**. Centralizar configurações em um único bloco no início garante que mudanças de parâmetros — como mover a janela de observação para um ano diferente — exijam alteração em um único lugar, não em dezenas de células espalhadas pelo notebook.

**Determinismo**. A semente fixa (`random_state = 42`) é aplicada em todas as operações estocásticas: divisão treino/teste, balanceamento, modelos com randomização interna. Isso garante que executar o notebook duas vezes na mesma máquina produza exatamente os mesmos números.

### Como faz

As janelas temporais são definidas como:

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `OBS_START` | 2017-01-01 | Início da janela de observação |
| `OBS_END` | 2018-01-01 | Fim da observação / início da predição |
| `PRED_END` | 2018-06-01 | Fim da janela de predição |

A escolha desses limites é deliberada: doze meses de observação capturam ciclos sazonais completos (Páscoa, dia das mães, dia dos namorados, Black Friday, Natal); seis meses de predição são suficientes para classificar com segurança um cliente como Churn em e-commerce, onde ciclos típicos de recompra ficam entre 3 e 4 meses.

### Resultado obtido

Ambiente configurado com 10 algoritmos disponíveis, paleta de cores consistente, warnings suprimidos.

---

## Seção 1 — Carregamento dos Dados

### O que faz

Carrega as seis tabelas relacionais do **Olist Brazilian E-Commerce Public Dataset** (Kaggle): pedidos, clientes, itens, pagamentos, avaliações e produtos.

### Por que faz assim

O dataset Olist é a maior base pública de e-commerce brasileiro disponível, com aproximadamente 100 mil pedidos reais entre 2016 e 2018. Sua estrutura relacional (seis tabelas conectadas por chaves primárias) reflete a realidade de qualquer e-commerce, o que torna o pipeline transferível para datasets corporativos.

A tabela `products_df` é fundamental: a categoria de cada produto **não está em `items_df`**, e sim em `products_df` (relacionada via `product_id`). Sem esse join intermediário, seria impossível calcular `n_categories` e `top_category` — duas features de engajamento importantes.

### Como faz

Leitura direta dos seis CSVs do diretório `data/` para `DataFrames` pandas.

### Resultado obtido

| Tabela | Linhas |
|--------|--------|
| Clientes | 99.441 |
| Pedidos | 99.441 |
| Itens | 112.650 |
| Pagamentos | 103.886 |
| Reviews | 99.224 |
| Produtos | 32.951 |

**73 categorias únicas** de produtos identificadas.

---

## Seção 2 — Limpeza e Split Temporal

### O que faz

Filtra os dados em duas dimensões: (i) mantém apenas pedidos com `status = 'delivered'` (transações efetivamente concluídas); (ii) divide os pedidos em duas janelas temporais — observação (até OBS_END) e predição (entre OBS_END e PRED_END).

### Por que faz assim

**Status `delivered`**: pedidos cancelados, devolvidos ou não entregues representam transações incompletas. Incluí-los introduziria ruído — um cliente cujo pedido foi cancelado pode ter alta intenção de compra mesmo sem transação efetiva. Manter apenas `delivered` garante que o histórico capturado reflete relações comerciais reais.

**Split temporal**: este é o elemento mais importante da metodologia. Sem janelas separadas, é fácil construir um modelo que "vê o futuro" — por exemplo, calcular a recência usando dados do mesmo período em que se está predizendo o churn. Esse fenômeno, conhecido como **data leakage** (KAUFMAN et al., 2012), produz métricas artificialmente altas no treino que desabam em produção. A janela dupla elimina esse risco estruturalmente: nenhuma feature pode ser computada com informação posterior a `OBS_END`.

### Como faz

```python
obs_orders  = pedidos com data < OBS_END
pred_orders = pedidos com OBS_END ≤ data < PRED_END
```

### Resultado obtido

| Conjunto | Pedidos |
|----------|---------|
| Pedidos entregues (total) | 96.478 |
| Janela de observação (2017) | 43.695 |
| Janela de predição (jan-jun 2018) | 34.174 |

---

## Seção 3 — Feature Engineering RFM

### O que faz

Calcula, para cada `customer_unique_id`, **onze features** que descrevem seu comportamento transacional na janela de observação. Em seguida, aplica o filtro `frequency ≥ 2`, mantendo apenas clientes com pelo menos duas compras.

### Por que faz assim

A metodologia **RFM (Recency, Frequency, Monetary)** é um dos frameworks mais consolidados de análise comportamental de clientes, com origem no marketing direto dos anos 1990 (HUGHES, 1994). As três dimensões capturam de forma objetiva o que importa em um relacionamento comercial: quando o cliente comprou pela última vez (recência), com que frequência ele compra (frequência) e quanto ele gasta (valor monetário).

Mas RFM puro é incompleto para predição de churn moderna. A literatura recente (MANZOOR et al., 2024) recomenda enriquecer com variáveis adicionais que capturem dimensões de engajamento, satisfação e padrão de pagamento. Por isso, foram adicionadas oito features complementares: ticket médio, tempo médio de entrega, nota média das avaliações, número médio de parcelas, frete médio, número de categorias distintas, categoria principal e estado do cliente.

**Por que filtrar `frequency ≥ 2`?** O Olist tem uma característica estrutural: cerca de 97% dos clientes compram uma única vez no período. Incluir esses one-time buyers no escopo de modelagem trivializa o problema — basta o modelo prever "Churn" para todos e ele acerta 97% dos casos sem ter aprendido nada. Compradores únicos também não têm histórico suficiente para caracterizar comportamento de recompra. Matuszelanski e Kopczewska (2022), no estudo mais rigoroso disponível sobre o Olist, recomendam explicitamente esse filtro. **Importante**: a Seção 4.1 validará empiricamente essa decisão, em vez de aceitá-la apenas por autoridade.

### Como faz

```python
# 11 features por customer_unique_id
recency_days       = OBS_END - data_ultimo_pedido
frequency          = COUNT(DISTINCT order_id)
monetary           = SUM(payment_value)
avg_ticket         = monetary / frequency
avg_delivery_days  = MEAN(delivered_date - purchase_date)
avg_review_score   = MEAN(review_score)
avg_installments   = MEAN(payment_installments)
avg_freight        = MEAN(freight_value)
n_categories       = COUNT(DISTINCT product_category_name)
top_category       = MODE(product_category_name) → Label Encoding
customer_state     = UF do cliente → Label Encoding
```

Tratamento de nulos via mediana para `review_score`, `payment_value` e `freight_value`. Categorias com `product_category_name` ausente (1,8% dos itens, comum em datasets reais) são marcadas como `'desconhecido'`.

### Resultado obtido

Join completo: 52.915 registros (pedido × cliente × item × pagamento × review).

Após agregação por cliente e filtro `frequency ≥ 2`: **1.178 clientes** com onze features cada.

---

## Seção 4 — Definição do Target

### O que faz

Cria a variável-alvo binária: `churn = 1` para clientes que **não** realizaram nenhuma compra na janela de predição (jan-jun/2018); `churn = 0` para os que realizaram pelo menos uma.

### Por que faz assim

A definição comportamental de churn — em oposição à definição contratual (cancelamento explícito) — é a única aplicável a e-commerce, onde não existe contrato a ser rescindido. Operacionalmente, é equivalente: cliente que não compra em seis meses é cliente perdido para fins de receita corrente.

A escolha do horizonte de seis meses equilibra dois critérios: (i) suficiência estatística para classificar com segurança (em ciclos de recompra de 3-4 meses, seis meses sem compra é sinal robusto); (ii) representatividade do horizonte preditivo — modelos de churn com janelas muito longas perdem acionabilidade operacional.

### Como faz

```python
active_future = clientes que compraram entre OBS_END e PRED_END
rfm['churn'] = 1 se cliente NÃO está em active_future, senão 0
```

### Resultado obtido

| Classe | Clientes | % |
|--------|----------|---|
| Churn (1) — não voltou | 1.128 | 95,8% |
| Ativo (0) — voltou | 50 | 4,2% |

O **desbalanceamento extremo** (~96/4) é a característica que mais define os desafios técnicos do problema. Boa parte do restante do pipeline lida explicitamente com as consequências desse desbalanceamento.

---

## Seção 4.1 — Validação Empírica do Filtro `frequency ≥ 2`

### O que faz

Treina dois modelos Random Forest de referência sobre dois conjuntos: (a) com o filtro adotado (`frequency ≥ 2`, 1.178 clientes) e (b) **sem** o filtro (`frequency ≥ 1`, mais de 42 mil clientes incluindo compradores únicos). Compara ROC-AUC e MCC para verificar se o filtro de fato melhora a capacidade preditiva.

### Por que faz assim

A decisão de excluir compradores únicos pode parecer intuitiva — eles não têm histórico de recompra. Mas decisões metodológicas não devem ser aceitas só por intuição em um trabalho científico. A questão concreta é: **incluir esses 41 mil clientes a mais ajudaria, ou apenas adicionaria ruído?**

Esta seção responde essa pergunta com evidência empírica. Se a inclusão dos compradores únicos melhorasse o modelo, o filtro seria injustificado. Se piorasse, o filtro está demonstrado como decisão correta.

### Como faz

Mesmas features RFM, mesmo modelo (Random Forest com hiperparâmetros padrão), mesmo split estratificado 80/20, mesma semente. Única diferença: o filtro de frequência.

### Resultado obtido

| Base | N clientes | Churn % | AUC | MCC |
|------|-----------|---------|-----|-----|
| freq ≥ 1 (sem filtro) | 42.136 | 98,9% | 0,6096 | 0,0000 |
| freq ≥ 2 (com filtro) | 1.178 | 95,8% | **0,7136** | **0,2099** |

**Conclusão**: o filtro **melhora** o desempenho — ROC-AUC sobe 0,104 pontos e o MCC vai de zero (classificador trivial) para 0,21 (discriminação real). Sem o filtro, o modelo essencialmente coloca todo mundo como Churn, acertando 98,9% por inércia. **A decisão de filtrar está validada empiricamente**, em consonância com a recomendação de Matuszelanski e Kopczewska (2022).

---

## Seção 5 — Análise Exploratória dos Dados (EDA)

### O que faz

Produz cinco visualizações que caracterizam o perfil comportamental dos clientes Ativos vs. Churn, identificando quais variáveis apresentam separação visual relevante entre as classes.

### Por que faz assim

A EDA cumpre três funções: (i) **hipóteses descritivas** — quais variáveis parecem discriminar bem as classes, orientando expectativas sobre quais features serão mais importantes no modelo; (ii) **verificação de sanidade** — confirmar que os dados refletem o que se espera de e-commerce (clientes Ativos com menor recência, maior monetário, etc.); (iii) **comunicação** — gráficos comparativos são frequentemente o material de apresentação mais convincente em contextos de negócio.

Cabe registrar que a EDA é **descritiva, não causal nem preditiva**. Diferenças observadas entre as classes na análise exploratória nem sempre se traduzem em alto poder preditivo no modelo final (devido a correlações entre features, controle estatístico, etc.). Esta distinção será importante na interpretação da feature `n_categories`.

### Como faz

Cinco visualizações:

1. **Distribuição de recência** por classe (histograma normalizado)
2. **Donut Churn vs. Ativo** (proporção das classes)
3. **Boxplot de frequência** por classe
4. **Review score médio** por classe
5. **Perfil comparativo multivariado** (seis distribuições lado a lado: recência, monetário, tempo de entrega, review score, ticket médio, número de categorias)
6. **Taxa de churn por categoria** (bar chart com média geral como referência)

### Resultado obtido

Padrões observados, consistentes com a literatura de churn em e-commerce:

- **Recência**: clientes Ativos concentrados nos primeiros 100 dias; Churn dispersos até 400 dias
- **Frequência**: Churn com mediana exatamente no piso (2 pedidos); Ativos com mediana superior e cauda longa (até 8 pedidos)
- **Review score**: diferença marginal entre as classes (Ativos: 4,25; Churn: 4,21), sugerindo que o problema **não é qualidade percebida do serviço, e sim engajamento**
- **Monetário**: Ativos concentrados em faixas superiores a R$ 400
- **Diversidade de categorias** (`n_categories`): Ativos compram em mais categorias distintas (média 1,70) que Churn (média 1,51)

A diferença em `n_categories` é o **insight de negócio mais saliente** da EDA: motiva a hipótese de que estratégias de cross-selling — estimular o cliente a explorar novas categorias — podem ter impacto na retenção. Esta hipótese será revisitada nas Seções 8 e 8.6 com perspectivas preditivas.

---

## Seção 6 — Pré-processamento e Balanceamento

### O que faz

Executa quatro operações em sequência: (i) Label Encoding das variáveis categóricas; (ii) split estratificado treino/teste 80/20; (iii) imputação de nulos por mediana e padronização das features, ajustadas **apenas no treino**; (iv) balanceamento combinado por undersampling da classe majoritária + oversampling sintético da minoritária com ruído gaussiano.

### Por que faz assim

**Label Encoding**: as categóricas (`top_category` com 73 níveis, `customer_state` com 27 UFs) precisam ser convertidas para numéricas. One-hot encoding criaria ~100 colunas adicionais, o que com apenas 1.178 amostras seria desastroso (maldição da dimensionalidade). Label Encoding é uma escolha conservadora que preserva a dimensionalidade — assume implicitamente ordem entre categorias, mas modelos baseados em árvores lidam bem com essa simplificação.

**`StratifiedShuffleSplit` 80/20**: preserva a proporção original de classes em ambos os conjuntos. Crítico em desbalanceamento extremo — um split aleatório poderia, por azar, deixar zero clientes Ativos no teste, inviabilizando avaliação.

**Fit do scaler apenas no treino**: ponto sutil mas central para evitar data leakage. Se a média e desvio padrão usados para padronizar fossem calculados sobre o dataset completo (incluindo o teste), o modelo "veria" estatísticas do conjunto de avaliação durante o treinamento. Ajustar apenas no treino e aplicar a transformação no teste preserva a separação rigorosa.

**Balanceamento combinado (undersampling + oversampling com ruído)**: quatro estratégias foram consideradas:

| Estratégia | Vantagem | Limitação |
|------------|----------|-----------|
| Sem balanceamento | Realismo da distribuição | Modelos ignoram a classe minoritária |
| Undersampling puro | Simples e rápido | Descarta informação da maioria; com 50 Ativos, treino diminuto (~100 amostras) |
| SMOTE (interpolação) | Consagrado na literatura | Em base pequena e esparsa, gera instâncias em regiões não representativas |
| **Under + Oversampling com ruído gaussiano (adotada)** | Equilibra as classes; perturbação pequena preserva a vizinhança real | Amplia a minoria a partir de base reduzida; exige validação (Seção 8.5) |

A estratégia adotada é mais conservadora que o SMOTE e mostrou-se superior em cenários de desbalanceamento extremo (HADDADI et al., 2024). A razão `3:1` (até 3 amostras Churn para cada Ativo) é um meio-termo entre balanceamento estrito (1:1, que descartaria muita informação Churn) e proporção original (que ignoraria a minoria).

**Balanceamento aplicado apenas no treino**: o conjunto de teste preserva a distribuição original (95,8% Churn), garantindo que a avaliação reflita condições reais de operação. Avaliar em conjunto balanceado superestima sistematicamente o desempenho.

### Como faz

```python
# Função balance_dataset
n_majoritaria_a_manter = min(len(maj), 3 × len(min))   # undersampling
n_sinteticas = n_majoritaria_a_manter - len(min)
amostras_sinteticas = amostras_min[escolhidas] + ruído_N(0, 0.05)  # oversampling
```

### Resultado obtido

| Conjunto | Tamanho | Distribuição |
|----------|---------|--------------|
| Treino original | 942 | 40 Ativos / 902 Churn |
| Treino balanceado | 240 | 120 Ativos / 120 Churn |
| Teste (hold-out) | 236 | 10 Ativos / 226 Churn (95,8% Churn — original) |

---

## Seção 7 — Benchmark de 10 Algoritmos

### O que faz

Treina **dez algoritmos de Machine Learning** sobre o conjunto de treino balanceado, com validação cruzada estratificada (5 folds) e avaliação no hold-out. Para cada modelo, otimiza o threshold de classificação por F1-Macro. Produz uma tabela comparativa multi-métrica completa.

### Por que faz assim

**Por que dez algoritmos?** O objetivo é cobrir as cinco principais famílias de classificadores aplicáveis ao problema, evitando viés de seleção:

- **Linear**: Regressão Logística (baseline interpretável)
- **Árvore única**: Decision Tree
- **Ensembles bagging**: Random Forest, Extra Trees
- **Ensembles boosting**: Gradient Boosting, HistGradient Boosting, AdaBoost
- **Vizinhança**: KNN
- **Probabilístico**: Naive Bayes
- **Margem máxima**: SVM

Limitar a um único algoritmo (ou a um pequeno subconjunto) introduziria viés de seleção — pode-se argumentar que outro algoritmo teria sido melhor. Com dez modelos, o resultado é defensável.

**Por que validação cruzada + hold-out?** A CV estima o desempenho médio em diferentes subdivisões do treino; o hold-out estima a generalização para dados nunca vistos. **A discrepância entre os dois é o indicador-chave de overfitting**: modelos com CV-AUC muito alta e AUC no hold-out muito baixa estão memorizando o conjunto de treino balanceado.

**Por que threshold tuning por F1-Macro?** Em desbalanceamento extremo, o threshold padrão de 0,50 quase sempre é subótimo — as probabilidades preditas concentram-se em valores altos para a classe majoritária. Buscar o threshold que maximiza F1-Macro (média harmônica balanceada entre Precisão e Recall por classe) é prática recomendada em FAWCETT (2006).

**Por que protocolo multi-métrica?** De la Cruz Huayanay, Bazán e Russo (2024) demonstraram que, em desbalanceamento extremo, **diferentes métricas podem indicar diferentes modelos como melhores**. Selecionar por uma única métrica (especialmente ROC-AUC) pode levar a escolhas frágeis. Avaliar por AUC, F1-Macro, MCC, G-Mean, Kappa e Precision-Recall simultaneamente é mais robusto.

### Como faz

Hiperparâmetros principais:

| Algoritmo | Hiperparâmetros |
|-----------|-----------------|
| Logistic Regression | `C=1.0, class_weight='balanced', max_iter=1000` |
| Decision Tree | `max_depth=8, class_weight='balanced'` |
| **Random Forest** | `n_estimators=300, max_depth=10, class_weight='balanced'` |
| Extra Trees | `n_estimators=300, max_depth=10, class_weight='balanced'` |
| Gradient Boosting | `n_estimators=200, max_depth=5, learning_rate=0.05` |
| HistGradient Boosting | `max_iter=300, max_depth=6, learning_rate=0.05, l2_regularization=0.1` |
| AdaBoost | `n_estimators=200, learning_rate=0.1` |
| KNN | `n_neighbors=7, weights='distance'` |
| Naive Bayes | (padrão) |
| SVM | `kernel='rbf', C=1.0, class_weight='balanced', probability=True` |

Todas as operações estocásticas com `random_state=42`.

### Resultado obtido

| Modelo | AUC | CV-AUC | F1-Macro | Prec. Ativo | Rec. Churn | Threshold |
|--------|-----|--------|----------|-------------|------------|-----------|
| **Random Forest** | **0,8235** | 0,9170 ± 0,043 | 0,6034 | 0,2857 | 0,9779 | 0,33 |
| HistGradient Boosting | 0,7735 | 0,8927 ± 0,050 | 0,6440 | 0,6667 | 0,9956 | 0,02 |
| Extra Trees | 0,7509 | 0,9431 ± 0,032 | 0,6115 | 0,2308 | 0,9558 | 0,43 |
| Gradient Boosting | 0,7416 | 0,8983 ± 0,055 | 0,5602 | 0,1304 | 0,9115 | 0,38 |
| Logistic Regression | 0,7111 | 0,7406 ± 0,021 | 0,6118 | 0,3333 | 0,9823 | 0,10 |
| SVM | 0,6960 | 0,8427 ± 0,045 | 0,5764 | 0,1818 | 0,9602 | 0,18 |
| Naive Bayes | 0,6520 | 0,6910 ± 0,030 | 0,5824 | 0,1667 | 0,9336 | 0,10 |
| Decision Tree | 0,6334 | 0,7573 ± 0,061 | 0,4964 | 0,0571 | 0,8540 | 0,01 |
| AdaBoost | 0,6144 | 0,8689 ± 0,054 | 0,5823 | 0,2000 | 0,9646 | 0,49 |
| KNN | 0,5232 | 0,9130 ± 0,043 | 0,5171 | 0,0714 | 0,9425 | 0,01 |

**Random Forest é o melhor modelo por ROC-AUC (0,8235)** com Recall Churn de 97,8%.

### Análise de overfitting (gap CV–hold-out)

| Modelo | Gap = CV-AUC − AUC | Status |
|--------|--------------------|--------|
| Random Forest | 0,094 | ✓ Aprovado (gap < 0,15) |
| HistGradient Boosting | 0,119 | ✓ Aprovado |
| Gradient Boosting | 0,157 | ⚠ Limite |
| Logistic Regression | 0,029 | ✓ Aprovado |
| SVM | 0,147 | ✓ Aprovado (no limite) |
| Naive Bayes | 0,042 | ✓ Aprovado |
| **Extra Trees** | **0,192** | ✗ **Descartado (overfitting)** |
| AdaBoost | 0,255 | ✗ Descartado |
| **KNN** | **0,390** | ✗ **Descartado (overfitting severo)** |
| Decision Tree | 0,124 | ✓ Aprovado |

**Extra Trees e KNN são descartados** pelo critério de overfitting (gap > 0,15). O Random Forest, com gap de apenas 0,094, está confortavelmente dentro do critério.

---

## Seção 7.1 — Análise de Robustez das Métricas

### O que faz

Calcula a **correlação de Spearman** entre os rankings dos 10 modelos quando ordenados por ROC-AUC, MCC, G-Mean e Kappa de Cohen. O objetivo é responder à pergunta: **a seleção pelo AUC é robusta a critérios alternativos de avaliação?**

### Por que faz assim

A análise responde diretamente à crítica metodológica de De la Cruz Huayanay, Bazán e Russo (2024): em desbalanceamento extremo, a ROC-AUC pode ser enganosa. Se os rankings por diferentes métricas concordarem fortemente (ρ próximo de 1), a seleção por AUC é robusta. Se discordarem, há risco de o modelo "vencedor por AUC" não ser o vencedor pelos critérios mais sensíveis ao desbalanceamento.

Esta análise não decide por si só qual modelo escolher — ela informa o grau de confiança que se pode ter na escolha por AUC. É uma verificação metodológica de segundo nível.

### Como faz

```python
spearmanr(rank_AUC, rank_MCC)
spearmanr(rank_AUC, rank_G-Mean)
spearmanr(rank_AUC, rank_Kappa)
```

### Resultado obtido

Spearman entre rankings (resultado da execução de referência):

- **AUC vs MCC**: ρ ≈ 0,59
- **AUC vs G-Mean**: ρ ≈ 0,59 (p ≈ 0,074)
- **AUC vs Kappa**: ρ ≈ 0,42

**Leitura**: associação positiva moderada com p-valor limítrofe. Há concordância parcial entre AUC e as métricas robustas — não há divergência crítica que invalide a escolha por AUC, mas também não há convergência forte que dispense a análise complementar. Esse resultado é coerente com o argumento de De la Cruz Huayanay et al. (2024) de que **AUC isolada é insuficiente** e justifica o portfólio multi-métrica adotado.

A análise é apresentada no TCC como motivação para a abordagem multi-métrica, não como crítica decisiva à seleção por AUC.

---

## Seção 7.2 — Visualizações do Benchmark

### O que faz

Produz sete gráficos comparativos: curvas ROC sobrepostas para todos os modelos, curvas Precision-Recall, matrizes de confusão lado a lado, heatmap das métricas, e barras comparativas de MCC, G-Mean e Kappa.

### Por que faz assim

Visualização é parte essencial de comunicação científica. Uma tabela com 10 modelos × 8 métricas é difícil de digerir; gráficos identificam padrões instantaneamente. As curvas ROC sobrepostas mostram qual modelo domina em diferentes regiões do espaço (alto recall vs. alta precisão); o heatmap revela quais modelos lideram em quais métricas.

### Como faz

Plots padronizados com a paleta de cores definida na Seção 0. Modelo vencedor destacado em cores diferenciadas.

### Resultado obtido

Gráficos confirmam visualmente o ranking quantitativo: Random Forest domina a curva ROC; HistGB tem alta Precisão Ativo; Extra Trees/KNN apresentam padrão visual claro de overfitting (CV alta, hold-out baixa).

---

## Seção 7.3 — Seleção do Modelo e Threshold Ótimo

### O que faz

Confirma a seleção do Random Forest como modelo de produção, otimiza o threshold por F1-Macro, e produz o classification report final no conjunto de teste.

### Por que faz assim

**Por que Random Forest?** Combinação de quatro fatores:

1. **Maior AUC entre os aprovados** (0,8235), com margem confortável sobre o segundo colocado (HistGB: 0,7735)
2. **Recall Churn de 97,8%** — captura praticamente todos os clientes em risco, característica prioritária quando o custo de não detectar um churner supera o custo de uma campanha de retenção desnecessária
3. **Gap CV–hold-out aceitável** (0,094) — não há indício de overfitting
4. **Importância de features nativa** via impureza de Gini, facilitando interpretabilidade

**Por que threshold 0,33?** A varredura de 0,01 a 0,99 com passo de 0,01 identifica 0,33 como o ponto que maximiza F1-Macro. Esse valor é operacionalmente plausível — não é tão baixo a ponto de sugerir probabilidades mal calibradas, e está abaixo do padrão 0,50, o que é esperado em desbalanceamento extremo (onde as probabilidades preditas concentram-se em valores altos para a classe majoritária).

### Como faz

```python
for thr in 0.01..0.99:
    f1 = f1_score(y_te, y_prob > thr, average='macro')
threshold_otimo = arg_max(f1)
```

### Resultado obtido

**Random Forest selecionado** com:

| Métrica | Valor |
|---------|-------|
| AUC | 0,8235 |
| Threshold ótimo | 0,33 |
| Acurácia | 94% |
| Precisão Ativo | 0,29 |
| Recall Ativo | 0,20 |
| Precisão Churn | 0,97 |
| Recall Churn | 0,98 |
| F1-Macro | 0,60 |

**Sobre o Recall Ativo baixo (0,20)**: com apenas 10 Ativos no hold-out, o modelo identifica corretamente 2 — limitação **estrutural do desbalanceamento**, não falha algorítmica. A Seção 9 reposiciona o modelo como **ranqueador de risco** (score contínuo → segmentação), não como classificador binário, mitigando essa limitação.

---

## Seção 8 — Interpretabilidade: Importância de Features (Gini)

### O que faz

Calcula a importância das 11 features no modelo Random Forest, usando o critério nativo de **impureza de Gini** — soma ponderada da redução de impureza que cada feature produz em cada split de cada árvore.

### Por que faz assim

A importância via Gini tem três vantagens em modelos baseados em árvores: (i) **determinística** dado o `random_state` — não depende de amostragem; (ii) **baixo custo computacional**; (iii) **nativa ao modelo** — não exige cálculo adicional via interpretadores externos.

Sua principal limitação é **não fornecer direção** do efeito (se a feature aumenta ou diminui a probabilidade de churn). Por isso, a análise é complementada pela Seção 8.6 com SHAP, que adiciona a dimensão direcional.

### Como faz

```python
importances = rf_model.feature_importances_  # Gini importance nativa do sklearn
ranking = sorted(features, key=importances, reverse=True)
```

### Resultado obtido

| Rank | Feature | Importância (Gini) |
|------|---------|---------------------|
| 1º | **frequency** | 22,95% |
| 2º | n_categories | 11,79% |
| 3º | recency_days | 11,55% |
| 4º | monetary | 8,10% |
| 5º | avg_installments | 7,30% |
| 6º | avg_delivery_days | 7,06% |
| 7º | avg_ticket | 6,93% |
| 8º | avg_freight | 6,54% |
| 9º | top_category | 6,54% |
| 10º | avg_review_score | 6,37% |
| 11º | customer_state | 4,87% |

**Top 3 — frequência, diversidade de categorias e recência — respondem por 46,3% da importância total**, e Top 5 por 61,7%.

**Achados-chave**:

- **`frequency` como principal driver** (22,95%) — confirma a intuição RFM clássica: clientes que compram mais vezes têm menor propensão ao churn.
- **`n_categories` como segundo driver** (11,79%) — sugere que a **diversidade de categorias é forte indicador de engajamento**. Este achado não está documentado nos estudos anteriores sobre o Olist (incluindo MATUSZELANSKI; KOPCZEWSKA, 2022). É o achado de negócio mais original do trabalho: motiva estratégias de cross-selling como mecanismo de retenção.
- **`recency_days` em terceiro** (11,55%), em empate técnico com `n_categories` (diferença < 0,3 ponto percentual). A ordenação entre 2º e 3º pode variar marginalmente entre execuções/versões de bibliotecas, mas a composição do trio dominante permanece estável.
- **`avg_review_score` apenas em 10º lugar** (6,37%) — confirma a observação da EDA: **o problema não é qualidade do serviço, e sim engajamento**.

---

## Seção 8.5 — Validação de Representatividade (PSI / KS-Test)

### O que faz

Aplica duas métricas complementares para verificar se o subconjunto de Churn que foi para o treino (após o undersampling 80/20) preserva a distribuição estatística da classe Churn completa:

- **PSI (Population Stability Index)** — quantifica deslocamento entre duas distribuições
- **KS-test (Kolmogorov-Smirnov, dois conjuntos)** — testa se duas amostras vêm da mesma distribuição

### Por que faz assim

Esta etapa responde a uma pergunta metodológica fundamental que a maioria dos estudos aplicados de churn **ignora**: 

> *O subconjunto de Churn usado no treino é representativo da classe Churn completa, ou aprendemos sobre uma versão distorcida?*

Se o undersampling aleatório, mesmo com semente fixa, tivesse selecionado uma subamostra enviesada (por exemplo, predominantemente clientes de SP, ou predominantemente compras de valor baixo), o modelo teria aprendido padrões que não generalizam. **A validação por PSI + KS é a forma rigorosa de descartar essa hipótese**.

Esta seção é uma das contribuições metodológicas que diferenciam este trabalho da literatura aplicada existente sobre o Olist.

**Critérios de interpretação**:

| Métrica | Faixa | Significado |
|---------|-------|-------------|
| PSI < 0,10 | Distribuições similares | Subconjunto representa bem a população |
| PSI 0,10–0,20 | Atenção | Algum deslocamento; investigar |
| PSI > 0,20 | Distorção significativa | Subconjunto não representa a população |
| KS p > 0,05 | Não rejeitar H₀ | Distribuições estatisticamente equivalentes |
| KS p ≤ 0,05 | Rejeitar H₀ | Distribuições significativamente diferentes |

### Como faz

```python
from scipy.stats import ks_2samp

for feature in features_numericas:
    psi = calcular_psi(churn_completo[feature], churn_treino[feature])
    ks_stat, p_valor = ks_2samp(churn_completo[feature], churn_treino[feature])
```

A função `calcular_psi` trata variáveis discretas de baixa cardinalidade (como `frequency`, com poucos valores únicos após o filtro) usando contagem por valor único em vez de quantis — refinamento técnico que evita degenerescência.

### Resultado obtido

Para as nove features numéricas, todos os valores de PSI ficaram **abaixo de 0,004** (muito abaixo do limiar de 0,10), e todos os p-valores do KS-test superaram **0,98**.

**Conclusão**: o undersampling aleatório com semente fixa **preservou a distribuição estatística da classe Churn**. O modelo aprendeu padrões da classe majoritária real, não de uma versão enviesada. **A representatividade está validada empiricamente.**

---

## Seção 8.6 — Análise SHAP

### O que faz

Calcula os **valores SHAP** (SHapley Additive exPlanations) via `TreeExplainer` para o modelo Random Forest, e produz quatro tipos de análise: summary plot (beeswarm), SHAP médio por classe real, dependence plots das top features, e waterfall plots de instâncias representativas.

### Por que faz assim

A importância via Gini (Seção 8) responde **quais features importam**. SHAP responde **em que direção cada feature influencia as predições e em quais limiares**. Essa dimensão direcional é essencial para tradução em recomendações de negócio.

Os valores SHAP, fundamentados na teoria dos jogos cooperativos de Shapley (1953), constituem o método de referência atual para explicabilidade local e global (LUNDBERG; LEE, 2017). O TreeSHAP (LUNDBERG et al., 2020) permite cálculo exato e eficiente para modelos baseados em árvores como Random Forest.

**Para cada predição**, SHAP quantifica a contribuição marginal de cada feature em relação à média do conjunto de treino. Valores positivos aumentam a probabilidade predita de churn; valores negativos a reduzem.

### Como faz

```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_te_sc)  # uma matriz por classe

# Quatro análises:
shap.summary_plot(shap_churn, X_te_sc_df)              # beeswarm
shap_medio_por_classe(shap_churn, y_te)                # SHAP médio Ativo vs Churn
shap.dependence_plot(top_features, shap_churn)         # limiares e interações
shap.waterfall(shap_values[cliente_tipico])            # decomposição individual
```

### Resultado obtido

**Concordância com Gini**: o ranking SHAP confirma o ranking Gini para as cinco principais features, com diferenças apenas em posições subsequentes. **Spearman ρ entre rankings ≈ 0,96** — rankings altamente concordantes.

**Direções dos efeitos** (lidos do beeswarm):

| Feature | Direção do efeito |
|---------|-------------------|
| `frequency` | **Valores baixos → aumentam P(Churn)** / valores altos → reduzem |
| `n_categories` | **Valores altos → reduzem P(Churn)** (mais categorias = menor risco) |
| `recency_days` | **Valores altos → aumentam P(Churn)** (mais dias sem comprar = mais risco) |
| `monetary` | Valores altos → reduzem P(Churn) (clientes que gastam mais retêm) |
| `avg_installments` | Valores altos → reduzem P(Churn) (parcelamento sinaliza relação de médio prazo) |

**Limiares não lineares**: o dependence plot revela que para `frequency`, o efeito sobre o risco é pronunciado entre 2 e 4 pedidos (cliente com 4+ pedidos tem risco estabilizado em patamar baixo); para `recency_days`, há aceleração do risco acima de 180 dias.

**Perfil do cliente que retorna** (leitura sob a ótica da retenção, invertendo as contribuições):

1. Compra em **mais categorias distintas** (n_categories elevado) — exploração ampla do catálogo
2. **Frequência ≥ 3 pedidos** no período de observação
3. **Recência baixa** — intervalos curtos entre transações
4. Maior **valor monetário acumulado**
5. **Utiliza parcelamento** em mais parcelas (sinal de relação de médio prazo)

---

## Seção 9 — Score de Churn e Segmentação de Risco

### O que faz

Aplica o Random Forest treinado à base completa (1.178 clientes), atribuindo a cada cliente um **score contínuo P(Churn)** entre 0 e 1. Segmenta os clientes em quatro faixas de risco com base em limiares fixos (0,25 / 0,50 / 0,75) e calcula o perfil médio de cada segmento.

### Por que faz assim

**Reposicionamento metodológico crítico**: o entregável principal **não é uma classificação binária** (Churn sim/não), e sim uma **probabilidade contínua que ordena os clientes por risco**. Essa mudança de perspectiva resolve a limitação do Recall Ativo baixo (Seção 7.3) — em vez de forçar uma decisão sim/não com support de 10 amostras, a segmentação em faixas oferece granularidade acionável: campanhas diferenciadas por nível de risco.

**Por que quatro faixas?** Equilibra granularidade (suficiente para diferenciar ações) e simplicidade (poucos segmentos para a equipe operacional gerenciar). Os limiares em quartis (0,25; 0,50; 0,75) são intuitivos e alinhados a Kotler e Keller (2016), que recomendam segmentação por risco em campanhas de retenção.

**Por que emojis**? Decisão de comunicação visual. Em apresentações para áreas de negócio, os emojis ⚪🟢🟡🟠🔴 transmitem instantaneamente o nível de urgência, sem exigir tradução técnica.

### Como faz

```python
rfm['churn_proba'] = rf_model.predict_proba(X_all)[:, 1]
rfm['risk_segment'] = cut(rfm['churn_proba'], bins=[0, 0.25, 0.50, 0.75, 1.001])
```

### Resultado obtido

| Segmento | Critério | Clientes | % | Recência média (dias) | Monetário médio (R$) | Review médio | P(Churn) média |
|----------|----------|----------|---|------------------------|------------------------|--------------|-----------------|
| 🟢 Baixo | P < 0,25 | 29 | 2,5% | 74,6 | 673,61 | 4,23 | 0,16 |
| 🟡 Médio | 0,25 ≤ P < 0,50 | 92 | 7,8% | 85,1 | 423,51 | 4,15 | 0,39 |
| 🟠 Alto | 0,50 ≤ P < 0,75 | 373 | 31,7% | 97,0 | 575,45 | 4,11 | 0,65 |
| 🔴 Crítico | P ≥ 0,75 | 684 | 58,1% | 146,6 | 408,03 | 4,27 | 0,85 |

**Observações sobre a distribuição**:

- **58% dos clientes em Crítico** — reflete o desbalanceamento extremo do problema (96% Churn na base); a maior parte dos clientes está em alto risco de não retornar
- **Segmento Baixo (29 clientes, 2,5%)** caracteriza-se pela **menor recência e maior valor monetário** — perfil clássico de clientes engajados de alto valor
- **Segmento Crítico (684 clientes, 58%)** tem a **maior recência (146 dias)** e o **menor monetário** — consistente com clientes prestes a abandonar

### Recomendações de ação por segmento

| Segmento | Ação recomendada | Princípio |
|----------|------------------|-----------|
| 🟢 Baixo | Cross-sell e manutenção de engajamento (programa de fidelidade) | Reforçar comportamento positivo |
| 🟡 Médio | Campanhas de reengajamento leve (newsletters, recomendação) | Investir moderado |
| 🟠 Alto | Ofertas e descontos personalizados; remarketing | Recuperar antes do abandono |
| 🔴 Crítico | Ação urgente: cupons de alto valor, contato direto | Última oportunidade |

Princípio econômico orientador: **gastar mais onde o risco é maior**, alinhado a Reichheld e Schefter (2000).

---

## Seção 10 — Exportação dos Resultados

### O que faz

Exporta um CSV com as predições de cada cliente para uso operacional: customer_id, features RFM resumidas, classe real, probabilidade predita, classe predita e segmento de risco.

### Por que faz assim

O CSV é a **interface entre o trabalho científico e a operação**. Mesmo que o modelo final seja servido via API, ter o arquivo CSV permite: (i) auditoria das predições; (ii) integração direta com ferramentas de marketing (Mailchimp, RD Station, planilhas); (iii) anexação ao TCC como artefato de evidência.

### Como faz

```python
rfm[['customer_unique_id', 'recency_days', 'frequency', 'monetary',
     'avg_review_score', 'top_category', 'customer_state',
     'churn', 'churn_proba', 'risk_segment']].to_csv('olist_churn_predictions.csv')
```

### Resultado obtido

Arquivo `olist_churn_predictions.csv` com **1.178 linhas × 10 colunas**.

---

## Síntese Final

### Os principais achados em uma página

**Sobre o modelo**:
- Random Forest selecionado para produção com ROC-AUC = 0,8235
- 7 dos 10 algoritmos avaliados com desempenho aceitável; Extra Trees e KNN descartados por overfitting
- Threshold ótimo = 0,33 (otimizado por F1-Macro)
- Recall Churn = 97,8% — captura praticamente todos os clientes em risco

**Sobre os drivers de churn** (Gini + SHAP):
- **Frequência de compras** é o principal driver (22,95%): clientes que compram mais retêm
- **Diversidade de categorias** (`n_categories`) e **recência** disputam o segundo lugar com importâncias muito próximas (~11,7% cada)
- **Satisfação declarada** (`avg_review_score`) tem importância secundária: o churn no Olist é um problema de **engajamento**, não de qualidade

**Sobre a metodologia**:
- O filtro `frequency ≥ 2` é **empiricamente validado**: melhora ROC-AUC em +0,104
- O undersampling aleatório com semente fixa é **estatisticamente representativo** (PSI < 0,004 para todas as features, KS p > 0,98)
- A divergência parcial entre rankings AUC e métricas robustas (Spearman ~0,59) **confirma a importância do portfólio multi-métrica**, em linha com De la Cruz Huayanay et al. (2024)
- Concordância forte (Spearman ~0,96) entre rankings de SHAP e Gini reforça a confiabilidade da análise de importância

**Sobre a aplicação de negócio**:
- 1.178 clientes scored e segmentados em quatro faixas de risco
- Estratégias diferenciadas de retenção por segmento, com ações urgentes para os 684 clientes do segmento Crítico
- A diversidade de categorias como driver sugere **cross-selling como mecanismo promissor de retenção**

### Por que este trabalho é diferente

| Diferencial | Onde aparece |
|-------------|--------------|
| Janela temporal dupla rigorosa | Seções 2, 3.2 |
| Validação empírica de decisões metodológicas | Seção 4.1 (filtro) |
| Critério explícito de descarte por overfitting | Seção 7 |
| Protocolo multi-métrica com análise de robustez | Seções 7, 7.1 |
| Validação de representatividade via PSI/KS | Seção 8.5 |
| Concordância Gini ↔ SHAP verificada | Seção 8.6 |
| Achado original: `n_categories` como driver de retenção | Seções 8, 8.6 |
| Reposicionamento como ranqueador de risco | Seção 9 |

### Limitações reconhecidas

1. **Base reduzida**: 1.178 clientes após filtro, com apenas 50 Ativos
2. **Período histórico**: dataset cobre 2016–2018; pode não refletir dinâmicas atuais do mercado
3. **MCC e Kappa moderados** (~0,31): consequência matemática do hold-out com apenas 10 Ativos, não falha algorítmica (LUQUE et al., 2019)
4. **Variabilidade entre versões de bibliotecas**: ainda com semente fixa, alterações de implementação no scikit-learn podem produzir variações marginais (~1 ponto percentual) nas métricas e na ordenação de features próximas

### Trabalhos futuros sugeridos

- Engenharia de features com dados de sessão e navegação
- Modelos de churn em tempo real via streaming
- Hiperparametrização sistemática (Optuna)
- Modelos de sobrevivência (Cox) para estimar tempo até o churn
- Calibração de probabilidades para boosting (Platt, isotônica)
- Validação experimental A/B das estratégias de cross-selling

---

## Glossário rápido

| Termo | Definição |
|-------|-----------|
| **Churn** | Cliente que não retorna no período de predição |
| **RFM** | Recency, Frequency, Monetary — framework clássico de análise comportamental |
| **Data leakage** | Vazamento de informação futura para o conjunto de treino |
| **Hold-out** | Conjunto separado para avaliação final, nunca visto durante o treino |
| **Validação cruzada (CV)** | Avaliação por divisão repetida do treino em sub-treino/sub-teste |
| **AUC (ROC-AUC)** | Área sob a curva ROC; mede capacidade discriminativa independente do threshold |
| **F1-Macro** | Média harmônica de Precisão e Recall, calculada por classe e depois mediada |
| **MCC** | Matthews Correlation Coefficient; métrica simétrica para classificação binária |
| **G-Mean** | Raiz quadrada do produto entre Sensibilidade e Especificidade |
| **Kappa de Cohen** | Concordância corrigida pelo acerto esperado ao acaso |
| **SHAP** | SHapley Additive exPlanations; método de explicabilidade baseado em teoria dos jogos |
| **PSI** | Population Stability Index; mede deslocamento entre distribuições |
| **KS-test** | Kolmogorov-Smirnov; teste de equivalência distribucional |
| **Overfitting** | Modelo memoriza o treino e generaliza mal para dados novos |
| **Threshold** | Limiar de decisão para converter probabilidade em classe |
| **Stratified split** | Divisão treino/teste que preserva a proporção das classes |

---

## Referências mobilizadas

Em ordem de relevância para o pipeline:

- **HUGHES (1994)** — Strategic Database Marketing → metodologia RFM (Seção 3)
- **KAUFMAN et al. (2012)** — Leakage in data mining → janelas temporais (Seções 2, 6)
- **MATUSZELANSKI & KOPCZEWSKA (2022)** — Estudo Olist → filtro de frequência (Seções 3, 4.1)
- **HE & GARCIA (2009)** — Imbalanced data → balanceamento (Seções 6, 8.5)
- **CHAWLA et al. (2002)** — SMOTE → comparação de estratégias (Seção 6)
- **HADDADI et al. (2024)** — Resampling para churn → estratégia adotada (Seção 6)
- **BREIMAN (2001)** — Random Forest → modelo de produção (Seções 2.3, 4.6)
- **FRIEDMAN (2001)** — Gradient Boosting → algoritmos comparados (Seção 7)
- **VAFEIADIS et al. (2015)** — Comparação de algoritmos para churn → desenho do benchmark (Seção 7)
- **FAWCETT (2006)** — ROC analysis → threshold tuning (Seção 7.3)
- **MATTHEWS (1975)** — MCC → métrica robusta (Seção 7.1)
- **CHICCO & JURMAN (2020)** — Vantagens do MCC → interpretação (Seção 7.1)
- **COHEN (1960)** — Kappa → métrica complementar (Seção 7.1)
- **SAITO & REHMSMEIER (2015)** — Curva PR em desbalanceamento → análise (Seção 7.2)
- **LUQUE et al. (2019)** — Impacto do desbalanceamento em métricas → interpretação MCC baixo (Seção 7)
- **DE LA CRUZ HUAYANAY et al. (2024)** — Métricas para classes desbalanceadas → motivação do protocolo multi-métrica (Seções 7, 7.1)
- **KUBAT & MATWIN (1997)** — One-sided selection → discussão metodológica (Seção 8.5)
- **SHAPLEY (1953)** — Teoria dos jogos → fundamento SHAP (Seção 8.6)
- **LUNDBERG & LEE (2017)** — Unified approach to interpreting → SHAP (Seção 8.6)
- **LUNDBERG et al. (2020)** — TreeSHAP → eficiência computacional (Seção 8.6)
- **REICHHELD & SCHEFTER (2000)** — E-loyalty → fundamento econômico (Seções 1, 9)
- **KOTLER & KELLER (2016)** — Marketing → segmentação (Seção 9)
- **HADDEN et al. (2007)** — Churn management → estado da arte (Seção 2.7)
- **IMANI et al. (2025)** — Revisão sistemática → contexto (Seção 2.7)
- **MANZOOR et al. (2024)** — Recomendações para praticantes → features complementares (Seção 3)
- **NESLIN et al. (2006)** — Defection detection → algoritmos (Seção 2.3)
- **PEDREGOSA et al. (2011)** — Scikit-learn → ferramentas (Seção 7)
- **TURBAN et al. (2018)** — E-commerce → contexto setorial (Seções 1, 2.1)
- **OLIST (2018)** — Dataset Olist no Kaggle → fonte de dados (Seção 1)
