# Guia Completo do Notebook `olist_churn_prediction.ipynb`

**Análise Preditiva de Churn em E-Commerce Brasileiro**
TCC, MBA em Inteligência Artificial e Big Data, ICMC/USP
Autor: Fabiano Ugulino. Orientadora: Profa. Dra. Cibele M. Russo

---

## Sobre este guia

Este documento explica, em ordem sequencial, o que o notebook faz, por que faz e o que ele encontra. O objetivo é que qualquer pessoa, mesmo sem familiaridade com Machine Learning aplicado a dados de e-commerce, consiga acompanhar o raciocínio do início ao fim, entender as decisões metodológicas e reproduzir o trabalho.

A estrutura segue exatamente a do notebook: Seções 0, 1, 1.1, 2, 3, 4, 4.1, 5, 6, 7, 7.1, 8, 8.5, 8.6, a análise de estabilidade que sucede o SHAP (numerada 4.6.1 no texto do TCC), 9 e 10, mais as duas figuras de fechamento e o sumário final. Cada bloco de célula é apresentado em quatro partes: **O que faz**, **Por que faz assim** (justificativa metodológica), **Como faz** (descrição técnica) e **Resultado obtido** (números da execução de referência, seed 42).

> Este guia foi revisado em agosto de 2026 para acompanhar uma reescrita relevante do notebook (correção de vazamento de informação no protocolo de validação cruzada e no ajuste do limiar de decisão, e substituição da alegação de "validação cruzada Gini/SHAP" por evidência de estabilidade entre partições). Números e afirmações anteriores a essa revisão não devem ser usados como referência.

---

## Visão geral da proposta

### O problema

A Olist é uma plataforma de marketplace que conecta pequenos varejistas brasileiros a grandes canais de venda. Como em todo e-commerce, uma parcela significativa dos clientes compra uma única vez e nunca mais retorna: fenômeno conhecido como **churn**. Antecipar quais clientes têm maior risco de abandono permite que a empresa direcione esforços de retenção (cupons, ofertas, contato) de forma proativa, em vez de reagir à perda já consumada.

### A pergunta de pesquisa

> Como identificar, de forma antecipada e com base apenas em dados transacionais históricos, quais clientes da Olist apresentam maior propensão a não voltar a comprar, e qual é o perfil do cliente que retorna?

### A abordagem em uma frase

Construir um modelo preditivo que, a partir de variáveis comportamentais de cada cliente (recência, frequência, valor, satisfação, padrão de pagamento, diversidade de categorias), estime a probabilidade de churn nos meses seguintes, com rigor metodológico para evitar vazamento de informação e validar a generalização.

### O que torna este trabalho diferente

- **Janela temporal dupla**, com observação e predição estritamente separadas, eliminando *data leakage*
- **Validação empírica do filtro de compradores únicos** (Seção 4.1): não é argumento puramente teórico, é demonstrado por ablação
- **Protocolo multi-métrica** (Seção 7 e 7.1): avalia AUC, F1-Macro, MCC, G-Mean, Kappa e Average Precision em conjunto, e testa a robustez da seleção por AUC via correlação de Spearman entre rankings, respondendo a uma limitação metodológica documentada por De la Cruz Huayanay, Bazán e Russo (2025)
- **Reamostragem e limiar sem vazamento** (revisão 2026-08): o balanceamento das classes ocorre dentro de cada fold da validação cruzada, e o limiar de decisão é derivado das probabilidades *out-of-fold* do treino, nunca do conjunto de teste
- **Validação de representatividade do balanceamento** (Seção 8.5, PSI e KS-test): etapa raramente conduzida em estudos aplicados, mas essencial para defender a validade interna dos resultados
- **Estabilidade da importância de features entre partições**, em vez de tratar a concordância Gini/SHAP como validação independente (os dois métodos leem a mesma estrutura de árvores)

### Bibliotecas usadas

```python
pandas, numpy           # manipulação e estatística
scikit-learn             # os dez algoritmos, métricas, pré-processamento
imbalanced-learn         # SMOTE, usado apenas na comparação de estratégias (Seção 6)
matplotlib, seaborn      # visualização
shap                     # explicabilidade (Seção 8.6)
scipy                    # correlação de Spearman, PSI e KS-test
```

---

## Seção 0, Imports e Configurações Globais

### O que faz

Carrega todas as bibliotecas necessárias, define a paleta de cores dos gráficos e fixa as constantes que delimitam as janelas temporais do estudo.

### Por que faz assim

Concentrar imports e constantes numa única célula garante que qualquer alteração de janela temporal se propague por todo o notebook a partir de um só ponto, evitando o erro comum de datas repetidas em várias células que saem de sincronia.

A semente fixa (`random_state = 42`) é aplicada em todas as operações estocásticas: divisão treino/teste, balanceamento, modelos com randomização interna. Isso garante que executar o notebook duas vezes na mesma máquina produza os mesmos números.

### Como faz

Constantes `OBS_START`, `OBS_END` e `PRED_END` marcam as janelas. Na prática, apenas `OBS_END` (2018-01-01) e `PRED_END` (2018-06-01) operam como filtros efetivos no pipeline (ver Seção 2); `OBS_START` é um marco nominal.

### Resultado obtido

```
✅ Todos os imports carregados com sucesso!
   Algoritmos disponíveis: LogisticRegression, DecisionTree, RandomForest,
   ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, KNN,
   GaussianNB, SVM
```

---

## Seção 1, Carregamento dos Dados

### O que faz

Lê as seis tabelas relacionais do **Olist Brazilian E-Commerce Public Dataset** (OLIST, 2018): pedidos, clientes, itens, pagamentos, avaliações e produtos.

### Por que faz assim

O dataset Olist é a maior base pública de e-commerce brasileiro disponível, com cerca de 100 mil pedidos reais entre 2016 e 2018. Sua estrutura relacional (seis tabelas conectadas por chaves) reflete a realidade de qualquer e-commerce corporativo.

### Uma armadilha central: `customer_id` não identifica pessoas

`customers_df` e `orders_df` têm exatamente 99.441 registros cada, porque no Olist o `customer_id` é gerado **por pedido**, não por pessoa: cada compra cria um novo `customer_id`. Quem identifica a pessoa ao longo do tempo é o `customer_unique_id`, com 96.096 valores distintos para 99.441 pedidos. A diferença são justamente os clientes recorrentes, objeto deste estudo. Por isso, **toda agregação por cliente neste notebook usa `customer_unique_id`**.

A categoria de cada produto também exige atenção: não está em `items_df`, mora em `products_df` e só é alcançável via `product_id`. Sem esse salto intermediário, a feature `n_categories` (um dos preditores mais fortes do modelo) não existiria.

### Como faz

Leitura direta dos seis CSVs do diretório `data/` para `DataFrames` pandas.

### Resultado obtido

| Tabela | Registros |
|--------|-----------|
| Clientes | 99.441 |
| Pedidos | 99.441 |
| Itens | 112.650 |
| Pagamentos | 103.886 |
| Reviews | 99.224 |
| Produtos | 32.951 |

**73 categorias únicas** de produtos identificadas.

---

## Seção 1.1, Caracterização da Base: Consultas Exploratórias

### O que faz

Antes de qualquer transformação, quantifica a base bruta em três blocos, respondendo a perguntas que fundamentam decisões metodológicas tomadas adiante.

### Por que faz assim

O objetivo é deixar rastreável a origem de cada número citado no TCC, com consultas simples e isoladas, em vez de um cálculo único difícil de auditar.

### Bloco A, Volumetria da base

Estabelece as contagens absolutas: pedidos, clientes únicos e a diferença entre ambos ("pedidos adicionais", que **não** equivale ao número de clientes recorrentes; um cliente com cinco pedidos contribui sozinho com quatro pedidos adicionais).

| Indicador | Valor |
|-----------|-------|
| Pedidos | 99.441 |
| Clientes únicos (`customer_unique_id`) | 96.096 |
| Pedidos adicionais | 3.345 |

### Bloco B, Recorrência de compra

Agrupa pedidos por `customer_unique_id` via join com `orders_df`, isola os clientes com mais de uma compra e descreve a distribuição do número de pedidos por cliente. Este bloco é a justificativa empírica do filtro `frequency >= 2` aplicado na Seção 4: com uma única transação não há intervalo entre pedidos, e recência e frequência tornam-se indistinguíveis do próprio tempo de vida do cliente.

| Qtd. pedidos | Qtd. clientes |
|---|---|
| 1 | 93.099 |
| 2 | 2.745 |
| 3 | 203 |
| 4 | 30 |
| 5 | 8 |
| 6 | 6 |
| 7 | 3 |
| 9 | 1 |
| 17 | 1 |

**Clientes recorrentes: 2.997** (96.096 menos os 93.099 com um único pedido). A cauda longa dessa distribuição, com a esmagadora maioria dos clientes aparecendo com um único pedido, é a origem estrutural do desbalanceamento tratado na Seção 6.

### Bloco C, Diversidade de categorias por cliente

Percorre o caminho de dois saltos (`orders_df` → `items_df` → `products_df`) e conta quantas categorias distintas cada cliente comprou. Constrói a evidência exploratória de `n_categories`, que se revelará entre as três features mais importantes do modelo. Nesta base ainda não filtrada, os clientes mais diversificados chegam a comprar em até **5 categorias** distintas.

> **Correlação, não causalidade.** A associação entre diversidade de categorias e retenção é correlacional. Comprar em várias categorias pode indicar engajamento preexistente em vez de causá-lo, ressalva mantida explicitamente no TCC.

---

## Seção 2, Limpeza, Conversão e Split Temporal

### O que faz

Converte colunas de data para `datetime`, filtra pedidos com `status == 'delivered'` e particiona os pedidos entregues em duas janelas temporais: observação e predição.

### Por que faz assim

**Status `delivered`**: pedidos cancelados ou em trânsito não representam consumo concluído e distorceriam o valor gasto.

**Split temporal**: este é o elemento metodológico mais importante do pipeline. Sem janelas separadas, é fácil construir um modelo que "vê o futuro", por exemplo calculando a recência com dados do mesmo período em que se está predizendo o churn. Esse fenômeno, conhecido como *data leakage* (KAUFMAN et al., 2012), produz métricas artificialmente altas no treino que desabam em produção. A literatura recente identifica este como um dos principais riscos metodológicos em estudos de churn (IMANI et al., 2025), e a maioria dos trabalhos sobre o dataset Olist não aplica essa separação rigorosa.

### Como faz

```python
obs_orders  = pedidos entregues com data < OBS_END   # 2018-01-01
pred_orders = pedidos entregues com OBS_END <= data < PRED_END   # 2018-06-01
```

A janela de observação **não** começa em `OBS_START`: o único filtro efetivo é `< OBS_END`, então ela cobre desde o primeiro pedido disponível na base (15/09/2016) até 31/12/2017, cerca de dezesseis meses. A constante `OBS_START` marca apenas o início nominal de 2017 e não é usada em nenhum filtro. A janela de predição vai de 01/01/2018 a 31/05/2018 (5 meses); como `PRED_END = 2018-06-01` entra com filtro estrito `<`, junho fica excluído.

### Resultado obtido

| Conjunto | Pedidos |
|----------|---------|
| Pedidos entregues (total) | 96.478 |
| Janela de observação (set/2016 a dez/2017) | 43.695 |
| Janela de predição (jan a mai/2018) | 34.174 |

A diferença entre o total entregue e a soma das duas janelas corresponde a pedidos entregues entre junho e outubro de 2018, fora do escopo de ambas as janelas.

---

## Seção 3, Feature Engineering, RFM + Variáveis Comportamentais

### O que faz

Une pedidos, clientes, itens, produtos, pagamentos e avaliações da janela de observação, calcula o tempo de entrega em dias, e agrega tudo por `customer_unique_id`, produzindo **onze features** por cliente.

### Por que faz assim

A metodologia **RFM (Recência, Frequência, Valor Monetário)** é um dos frameworks mais consolidados de análise comportamental de clientes, com origem no marketing direto dos anos 1990 (HUGHES, 1994). As três dimensões capturam de forma objetiva o que importa em um relacionamento comercial: quando o cliente comprou pela última vez, com que frequência ele compra e quanto ele gasta.

RFM puro é incompleto para predição de churn moderna. A literatura recente (MANZOOR et al., 2024) recomenda enriquecer com variáveis adicionais que capturem engajamento, satisfação e padrão de pagamento. Por isso, foram adicionadas oito features complementares: ticket médio, tempo médio de entrega, nota média das avaliações, número médio de parcelas, frete médio, número de categorias distintas, categoria principal e estado do cliente.

O ponto mais delicado do cálculo é `monetary`. O join intermediário replica o pagamento uma vez por item (43.695 pedidos viram 52.915 linhas): somar `payment_value` diretamente inflaria o gasto de quem comprou vários produtos no mesmo pedido. A solução reconstrói o valor de cada pedido a partir de `payments_df` e agrega por cliente contando cada pedido uma única vez, garantindo que `avg_ticket = monetary / frequency` seja de fato o ticket médio por pedido.

### Como faz

```python
# 11 features por customer_unique_id
recency_days       = OBS_END - data_ultimo_pedido
frequency          = COUNT(DISTINCT order_id)
monetary           = SUM(payment_value por pedido, sem duplicar por item)
avg_ticket         = monetary / frequency
avg_delivery_days  = MEAN(delivered_date - purchase_date), clip(lower=0)
avg_review_score   = MEAN(review_score)
avg_installments   = MEAN(payment_installments)
avg_freight        = MEAN(freight_value)
n_categories       = COUNT(DISTINCT product_category_name)
top_category       = MODE(product_category_name)
customer_state     = UF do cliente
```

Nulos são tratados por critério específico de cada variável, não por um valor único: mediana para avaliações e pagamentos, zero para frete (ausência significa frete grátis) e `1` para parcelas (à vista).

### Resultado obtido

Join completo: 52.915 linhas (uma por item de pedido), 20 colunas. Categorias conhecidas em 51.972 de 52.915 itens (98,2%); o restante fica marcado como `desconhecido`. Sem nulos remanescentes após a imputação.

Agregando por cliente, o resultado é a base **ainda sem o filtro de recorrência**: **42.395 clientes únicos**, com onze features cada. Esta é a base `rfm_unfiltered`, reutilizada na validação empírica da Seção 4.1.

---

## Seção 4, Definição do Target, Churn por Janela de Tempo

### O que faz

Cria a variável-alvo binária (`churn = 1` para clientes que **não** compraram na janela de predição, `churn = 0` para os que compraram pelo menos uma vez) e aplica o filtro `frequency >= 2`, restringindo a base a clientes com histórico de recompra.

### Por que faz assim

A definição comportamental de churn, em oposição à contratual (cancelamento explícito), é a única aplicável a e-commerce, onde não existe contrato a ser rescindido: o abandono é inferido pela ausência de nova compra em um horizonte predefinido (NESLIN et al., 2006; HADDEN et al., 2007).

**Por que filtrar `frequency >= 2`?** Compradores únicos seriam classificados como Churn de forma trivial, sem que o modelo aprendesse padrão comportamental real, e não têm histórico suficiente para caracterizar recompra. Matuszelański e Kopczewska (2022), no estudo mais rigoroso disponível sobre o Olist, recomendam esse filtro. A base `rfm_unfiltered` (42.395 clientes) é preservada justamente para permitir a validação empírica que se segue na Seção 4.1, em vez de aceitar a decisão apenas por autoridade.

### Como faz

```python
active_future = clientes que compraram entre OBS_END e PRED_END
rfm['churn']  = 1 se cliente NÃO está em active_future, senão 0
rfm           = rfm[rfm['frequency'] >= 2]
```

### Resultado obtido

Após o filtro, a base de modelagem tem **1.178 clientes**:

| Classe | Clientes | % |
|--------|----------|---|
| Churn (1), não voltou | 1.128 | 95,8% |
| Ativo (0), voltou | 50 | 4,2% |

O **desbalanceamento extremo** (~96/4) é a característica que mais define os desafios técnicos do problema e condiciona todo o restante do pipeline.

---

## Seção 4.1, Validação Empírica da Exclusão de Compradores Únicos

### O que faz

Roda um experimento de ablação: treina o mesmo Random Forest de referência, com o mesmo protocolo de avaliação, em duas configurações (com e sem o filtro `frequency >= 2`), e compara AUC e MCC.

### Por que faz assim

A exclusão de compradores únicos pode parecer intuitiva, mas decisões metodológicas de peso não devem ser aceitas apenas por intuição num trabalho científico. A questão concreta é: incluir 41 mil clientes a mais ajudaria o modelo, ou apenas adicionaria ruído? Esta seção responde com evidência empírica, em vez de apenas justificar em texto. É também a resposta direta a um questionamento recorrente em bancas: "por que descartar cerca de 96% dos clientes?"

Nesta célula também são definidas as funções centrais reaproveitadas em todo o notebook: `balance_dataset` (undersampling + oversampling com ruído gaussiano), `probabilidades_out_of_fold` (validação cruzada sem vazamento) e `limiar_out_of_fold` (limiar derivado apenas do treino). Ver Seção 6 e 7 para o detalhamento de cada uma.

### Como faz

Mesmas features, mesmo modelo (Random Forest, hiperparâmetros padrão de produção), mesmo split estratificado 80/20, mesma semente. O limiar de decisão de cada configuração é obtido das probabilidades *out-of-fold* do próprio treino, nunca do conjunto de teste.

### Resultado obtido

| Config | Clientes | Churn % | Ativos no teste | AUC | MCC | CV-AUC | Limiar |
|--------|----------|---------|------------------|-----|-----|--------|--------|
| **Adotada** (freq >= 2) | 1.178 | 95,8% | 10 | **0,8106** | **0,1931** | 0,6814 | 0,38 |
| Alternativa (freq >= 1, inclui únicos) | 42.395 | 98,9% | 94 | 0,5231 | 0,0400 | 0,5817 | 0,31 |

Incluir compradores únicos derruba o AUC em **0,2875 ponto** (aproximando o modelo do acaso) e o MCC em **79%**. **A exclusão de compradores únicos está empiricamente validada**: longe de ser uma escolha de conveniência, é uma decisão que melhora mensuravelmente o poder discriminativo do modelo.

---

## Seção 5, Análise Exploratória dos Dados (EDA)

### O que faz

Calcula estatísticas descritivas por classe e produz seis visualizações que caracterizam o perfil comportamental de clientes Ativos versus Churn.

### Por que faz assim

A EDA cumpre três funções: gerar hipóteses descritivas sobre quais variáveis discriminam bem as classes; verificar a sanidade dos dados; e fornecer material de comunicação para contextos de negócio. É importante registrar que a EDA é descritiva, não causal nem preditiva: diferenças observadas aqui nem sempre se traduzem em alto poder preditivo no modelo final, devido a correlações entre features e efeitos de interação.

### Como faz

Painel 2x2 (recência, proporção de classes, boxplot de frequência, review score), painel 2x3 do perfil multivariado (seis distribuições sobrepostas, com a média de cada grupo marcada) e um gráfico de taxa de churn por categoria de produto.

### Resultado obtido

Estatísticas por classe na coorte de 1.178 clientes:

| Variável | Ativo (média) | Churn (média) | Ativo (mediana) | Churn (mediana) |
|---|---|---|---|---|
| Recência (dias) | 84,10 | 126,08 | 63,50 | 109,00 |
| Frequência (pedidos) | 2,56 | 2,08 | 2,00 | 2,00 |
| Valor monetário (R$) | 372,25 | 299,12 | 226,56 | 213,24 |
| Ticket médio (R$) | 150,70 | 143,97 | 108,40 | 104,80 |
| Categorias distintas | 1,80 | 1,54 | 2,00 | 2,00 |
| Review score médio | 4,25 | 4,21 | 5,00 | 5,00 |
| Tempo de entrega (dias) | 10,94 | 11,95 | 10,00 | 11,00 |

66,0% dos Ativos têm recência de até 100 dias. A frequência máxima observada é 8 pedidos entre os Ativos e 6 entre os Churn. Nesta coorte já filtrada (`frequency >= 2`), o número de categorias distintas não ultrapassa 4 em nenhuma das duas classes.

**Leituras principais**:

- **Frequência**: Churn com mediana no piso (2 pedidos); Ativos com cauda mais longa (até 8 pedidos)
- **Review score**: diferença marginal entre as classes (4,25 contra 4,21), sugerindo que o problema **não é qualidade percebida do serviço, e sim engajamento**
- **Diversidade de categorias**: Ativos compram em mais categorias (média 1,80) que Churn (média 1,54); é o insight de negócio mais saliente da EDA, revisitado com evidência preditiva nas Seções 8, 8.6 e na análise de estabilidade

---

## Seção 6, Pré-processamento e Balanceamento de Classes

### O que faz

Executa quatro operações em sequência: (i) Label Encoding das variáveis categóricas; (ii) split estratificado treino/teste 80/20; (iii) imputação de nulos por mediana e padronização, ajustadas apenas no treino; (iv) balanceamento do treino por undersampling da classe majoritária combinado a oversampling sintético da minoritária com ruído gaussiano.

### Por que faz assim

**Label Encoding**: as categóricas (`top_category` com dezenas de níveis, `customer_state` com 27 UFs) precisam virar numéricas. One-hot encoding criaria dezenas de colunas adicionais, o que com apenas 1.178 amostras seria desastroso (maldição da dimensionalidade). Label Encoding preserva a dimensionalidade; modelos baseados em árvores lidam bem com essa simplificação.

**`StratifiedShuffleSplit` 80/20**: preserva a proporção original de classes em ambos os conjuntos. Em desbalanceamento extremo, um split aleatório simples poderia deixar o teste sem nenhum cliente Ativo, inviabilizando a avaliação.

**Ajuste do imputador e do scaler apenas no treino**: se a mediana e o desvio padrão fossem calculados sobre o dataset completo, informação do teste vazaria para o treino (KAUFMAN et al., 2012).

**Balanceamento combinado (undersampling + oversampling com ruído gaussiano)**: a função `balance_dataset`, definida na Seção 4.1, não é SMOTE. Não há interpolação entre vizinhos; cada instância sintética é uma cópia de uma observação real da classe Ativo somada a ruído `N(0; 0,05)` sobre as features já padronizadas, o que preserva a vizinhança original em vez de criar pontos em regiões não observadas do espaço de features. A razão de undersampling é `3:1` (até 3 amostras Churn por amostra Ativo), meio-termo entre balanceamento estrito (que descartaria muita informação Churn) e proporção original (que ignoraria a minoria). Essa estratégia mais conservadora que o SMOTE mostrou-se vantajosa em desbalanceamento extremo (HADDADI et al., 2024), e a Seção 6.1 (a seguir) compara as quatro alternativas de forma empírica.

**Balanceamento aplicado apenas ao treino**: o teste preserva a distribuição original (95,8% Churn), para que a avaliação reflita as condições reais de operação.

### Como faz

```python
n_maj_keep = min(len(maj), RATIO_MAJ * len(min))     # undersampling, RATIO_MAJ = 3
n_synth    = max(0, n_maj_keep - len(min))
X_synth    = amostras_min[escolhidas] + N(0, 0.05)    # oversampling
```

### Resultado obtido

| Conjunto | Tamanho | Distribuição |
|----------|---------|--------------|
| Treino original | 942 | 40 Ativos / 902 Churn |
| Treino balanceado | 240 | 120 Ativos / 120 Churn |
| Teste (hold-out) | 236 | 10 Ativos / 226 Churn (95,8% Churn, original) |

Onze features ao todo: 9 numéricas e 2 categóricas codificadas.

### Comparação empírica das estratégias de balanceamento

Avalia quatro alternativas (sem balanceamento, undersampling puro 1:1, SMOTE e a combinação adotada under 3:1 + ruído gaussiano), primeiro num split de referência e depois em **trinta divisões independentes**, com reamostragem sempre feita dentro do fold e limiar sempre derivado do treino.

Com apenas dez clientes Ativos no teste, um único split pode favorecer qualquer estratégia por acaso. Repetir trinta vezes e reportar média com desvio padrão mostra se a vantagem observada é real ou coincidência.

**Split de referência (random_state=42):**

| Estratégia | N treino | AUC | CV-AUC | MCC | Kappa | F1-Macro | Rec-Ativo | Rec-Churn | Limiar |
|---|---|---|---|---|---|---|---|---|---|
| Sem balanceamento | 942 | 0,8018 | 0,6017 | 0,2983 | 0,2680 | 0,6319 | 0,20 | 0,9912 | 0,75 |
| Undersampling puro (1:1) | 80 | 0,7451 | 0,6080 | 0,1943 | 0,1884 | 0,5930 | 0,30 | 0,9425 | 0,27 |
| SMOTE | 1.804 | 0,7327 | 0,5928 | 0,1253 | 0,1233 | 0,5610 | 0,20 | 0,9469 | 0,44 |
| **Under 3:1 + gaussiano (adotada)** | 240 | **0,8106** | 0,6814 | 0,1931 | 0,1918 | 0,5957 | 0,20 | 0,9735 | 0,38 |

**Robustez em 30 splits estratificados independentes (média ± desvio padrão):**

| Estratégia | AUC média | MCC médio | Kappa médio | AUC min-max |
|---|---|---|---|---|
| Sem balanceamento | 0,6543 ± 0,0778 | 0,1168 ± 0,1167 | 0,1031 ± 0,1004 | 0,462 a 0,800 |
| Undersampling puro (1:1) | 0,6510 ± 0,0769 | 0,0439 ± 0,0765 | 0,0392 ± 0,0689 | 0,501 a 0,802 |
| SMOTE | 0,6615 ± 0,0660 | 0,1060 ± 0,0761 | 0,0895 ± 0,0696 | 0,539 a 0,810 |
| **Under 3:1 + gaussiano** | **0,6686 ± 0,0831** | 0,1140 ± 0,0804 | 0,1053 ± 0,0726 | 0,500 a 0,822 |

A estratégia adotada lidera em AUC nas duas avaliações, mas o resultado é reportado com honestidade quanto aos limites: **as diferenças entre estratégias cabem dentro de um desvio padrão**, o que indica tendência, não superioridade estatisticamente estabelecida.

---

## Seção 7, Comparação de Algoritmos, Benchmark Completo

### O que faz

Treina **dez algoritmos de Machine Learning**, todos com validação cruzada estratificada de 5 folds e reamostragem executada dentro de cada fold, e avalia cada um no hold-out por sete métricas. Otimiza o ponto de corte individualmente e monta o leaderboard comparativo.

### Por que dez algoritmos

Cobrir as principais famílias de classificadores tabulares evita viés de seleção: linear (Regressão Logística), árvore única (Decision Tree), ensembles de bagging (Random Forest, Extra Trees), ensembles de boosting (Gradient Boosting, HistGradient Boosting, AdaBoost), vizinhança (KNN), probabilístico (Naive Bayes) e margem máxima (SVM). Limitar a um único algoritmo permitiria argumentar que outro teria sido melhor; com dez, a comparação é defensável.

### Por que o protocolo de validação foi corrigido (revisão 2026-08)

A versão original deste pipeline continha dois vieses, corrigidos após revisão da orientadora:

1. **Reamostragem fora do fold**: a validação cruzada rodava sobre um conjunto já balanceado, espalhando instâncias sintéticas geradas da mesma observação original entre folds distintos e inflando a estimativa (vazamento de informação entre treino e validação). Agora, imputação, padronização e balanceamento são ajustados **dentro** de cada fold, exclusivamente sobre a porção de ajuste; a porção de validação permanece na distribuição original de classes.
2. **Limiar escolhido no teste**: o ponto de corte era escolhido maximizando F1-Macro diretamente contra `y_te`, usando o conjunto de teste duas vezes (uma para ajustar o limiar, outra para reportar a métrica), o que produzia viés otimista em todas as métricas dependentes de limiar. Agora, o limiar é derivado das **probabilidades out-of-fold do treino** e congelado antes de qualquer contato com o hold-out. A coluna `CV-AUC vazada` do leaderboard reproduz deliberadamente o procedimento anterior (validação cruzada sobre o conjunto já balanceado), apenas para dimensionar o efeito da correção, e não deve ser lida como resultado.

### Por que protocolo multi-métrica

Em contextos desbalanceados a acurácia é enganosa (prever sempre Churn dá cerca de 96% de acurácia sem nenhum poder preditivo). De la Cruz Huayanay, Bazán e Russo (2025), em estudo publicado na *Computational Statistics*, demonstraram via simulação extensiva que **AUC e acurácia têm desempenho inadequado** para seleção de modelos em dados desbalanceados, enquanto **MCC, G-Mean e Kappa de Cohen** são consistentemente adequadas. Este trabalho mantém a AUC por comparabilidade, mas reporta as três métricas recomendadas lado a lado.

### Sobre o critério de exclusão por overfitting

A diferença `Δ = |CV-AUC - AUC hold-out|` é **verificada e reportada, mas não usada para descartar modelos**. Nenhum modelo foi eliminado do leaderboard por esse critério: o maior Δ observado é 0,1347 (Decision Tree), seguido de 0,1293 (Random Forest), ambos dentro do limite de referência `GAP_MAX = 0,15` definido no notebook. Esta é uma mudança em relação a versões anteriores do trabalho, que descartavam modelos com gap elevado.

### Sobre o limiar ótimo

O limiar de cada modelo maximiza o **Kappa de Cohen** (`CRITERIO_LIMIAR = 'kappa'`), critério também adotado por De la Cruz Huayanay, Bazán e Russo (2025) para fixar o ponto de corte. A varredura testa 200 valores entre 0,01 e 0,99. A variante que maximiza F1-Macro é calculada em paralelo como análise de sensibilidade: as duas convergem em oito dos dez modelos, divergindo apenas no SVM e na Árvore de Decisão.

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

### Resultado obtido, leaderboard (ordenado por AUC)

| Modelo | AUC | CV-AUC | Δ CV-holdout | F1-Macro | MCC | G-Mean | Kappa | Prec-Ativo | Rec-Ativo | Rec-Churn | Limiar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **0,8106** | 0,6814 | 0,1293 | 0,5957 | 0,1931 | 0,4412 | 0,1918 | 0,2500 | 0,20 | 0,9735 | 0,38 |
| HistGradient Boosting | 0,7615 | 0,7363 | 0,0252 | 0,6049 | 0,2143 | 0,5342 | 0,2110 | 0,2143 | 0,30 | 0,9513 | 0,18 |
| Extra Trees | 0,7491 | 0,6777 | 0,0714 | 0,5649 | 0,1639 | 0,3148 | 0,1370 | 0,3333 | 0,10 | 0,9912 | 0,34 |
| Logistic Regression | 0,7075 | 0,6326 | 0,0749 | 0,6440 | 0,3517 | 0,4462 | 0,2939 | 0,6667 | 0,20 | 0,9956 | 0,12 |
| SVM | 0,7035 | 0,6238 | 0,0798 | 0,4920 | 0,0250 | 0,4111 | 0,0197 | 0,0541 | 0,20 | 0,8451 | 0,37 |
| Gradient Boosting | 0,6642 | 0,6695 | 0,0053 | 0,5339 | 0,0679 | 0,3106 | 0,0678 | 0,1111 | 0,10 | 0,9646 | 0,08 |
| Naive Bayes | 0,6405 | 0,6390 | 0,0015 | 0,5887 | 0,1778 | 0,4402 | 0,1775 | 0,2222 | 0,20 | 0,9690 | 0,01 |
| AdaBoost | 0,5544 | 0,5372 | 0,0172 | 0,5725 | 0,2100 | 0,3155 | 0,1547 | 0,5000 | 0,10 | 0,9956 | 0,47 |
| KNN | 0,5336 | 0,5913 | 0,0577 | 0,5143 | 0,0314 | 0,3063 | 0,0307 | 0,0667 | 0,10 | 0,9381 | 0,09 |
| Decision Tree | 0,4772 | 0,6119 | 0,1347 | 0,4592 | 0,0178 | 0,4708 | 0,0112 | 0,0484 | 0,30 | 0,7389 | 0,61 |

**Random Forest lidera por ROC-AUC** (0,8106), mas não domina em todas as métricas: a Regressão Logística tem o maior MCC (0,3517) e Kappa (0,2939), e o HistGradient Boosting o maior G-Mean (0,5342). Essa divergência é justamente o achado que a Seção 7.1 investiga formalmente.

### Curvas ROC, matriz de confusão, ponto de corte e Average Precision

As mesmas dez avaliações também são visualizadas em curvas ROC sobrepostas, barras comparativas de AUC/MCC/G-Mean/Kappa e um painel de matrizes de confusão. Para o Random Forest, a matriz de confusão no hold-out mostra 2 Ativos corretos, 8 Ativos perdidos, 6 Churn perdidos e 220 Churn corretos (consistentes com precisão-Ativo 0,25 e recall-Churn 0,9735 da tabela acima).

A distribuição das probabilidades do Random Forest no hold-out tem mediana 0,7787 (mín. 0,1121, máx. 0,9714); no treino *out-of-fold*, mediana 0,7721. O limiar adotado (0,38, derivado do treino) produz Kappa de 0,2049 no próprio treino out-of-fold e 0,1918 no hold-out; se o limiar fosse escolhido diretamente no teste, o Kappa chegaria a 0,2254 no limiar 0,34, uma diferença de 0,0336 que mede exatamente o viés otimista que o procedimento antigo incorporava.

Classification report do Random Forest no hold-out (limiar 0,38):

```
              precision    recall  f1-score   support
       Ativo       0.25      0.20      0.22        10
       Churn       0.96      0.97      0.97       226
    accuracy                           0.94       236
```

Sobre a curva Precision-Recall: a Average Precision (AP) da classe Ativo, que é a métrica sensível ao ranqueamento em desbalanceamento severo, varia por modelo, com o Random Forest em 0,1494 contra um baseline de 0,042 (lift de 3,53x). O HistGradient Boosting tem AP-Ativo mais alta (0,2559, lift 6,04x) e a Naive Bayes lidera esse indicador isoladamente (0,2701, lift 6,37x), embora tenha a AUC mais baixa entre os modelos aprovados: outra evidência da divergência entre critérios explorada a seguir.

---

## Seção 7.1, Análise das Métricas Robustas (MCC, G-Mean, Kappa)

### O que faz

Ordena os dez modelos por AUC, MCC, G-Mean e Kappa, e mede o grau de concordância entre os rankings pela correlação de Spearman.

### Por que faz assim

Responde a uma pergunta raramente feita: escolher o modelo por AUC leva ao mesmo resultado que escolher por MCC, G-Mean ou Kappa? Esta análise responde diretamente à recomendação de De la Cruz Huayanay, Bazán e Russo (2025) de nunca usar uma métrica isolada em desbalanceamento extremo, e informa o grau de confiança que se pode ter na escolha do Random Forest por AUC.

### Como faz

```python
spearmanr(rank_AUC, rank_MCC)
spearmanr(rank_AUC, rank_G-Mean)
spearmanr(rank_AUC, rank_Kappa)
```

### Resultado obtido

- **AUC vs MCC**: rho = +0,5515, p = 0,0984 (não significativa a 5%)
- **AUC vs G-Mean**: rho = +0,2970, p = 0,4047 (não significativa a 5%)
- **AUC vs Kappa**: rho = +0,6485, p = 0,0425 (significativa a 5%, a única das três)

Os quatro critérios elegem **três vencedores distintos**: Random Forest pela AUC, Regressão Logística por MCC e por Kappa, HistGradient Boosting por G-Mean e por Recall-Ativo.

**Conclusão do notebook**: os rankings são **divergentes** em pelo menos uma métrica, e a seleção por ROC-AUC **não é confirmada de forma unânime** pelas métricas robustas. Isso não invalida a escolha do Random Forest (justificada na próxima seção por um conjunto mais amplo de critérios), mas confirma empiricamente a recomendação de nunca decidir por uma métrica isolada em desbalanceamento extremo. A ressalva do próprio notebook: os coeficientes são estimados sobre apenas dez modelos, e devem ser lidos como indicativos da tendência observada, não como estimativa populacional precisa.

---

## Seção 8, Interpretabilidade, Feature Importance (Gini)

### O que faz

Extrai do Random Forest treinado a importância de cada uma das onze features via impureza de Gini: a soma ponderada da redução de impureza que a feature produz em todos os splits de todas as árvores (BREIMAN, 2001).

### Por que faz assim

A importância via Gini é determinística, computacionalmente barata e nativa ao modelo, sem exigir cálculo adicional. Sua limitação principal é não informar a direção do efeito (se a feature aumenta ou reduz a probabilidade de churn), papel que a Seção 8.6 cumpre com SHAP, e tende a favorecer variáveis contínuas de alta cardinalidade, o que pode subestimar categóricas como `top_category`.

### Como faz

```python
importances = rf_model.feature_importances_
ranking = sorted(features, key=importances, reverse=True)
```

### Resultado obtido

| Rank | Feature | Importância (Gini) |
|------|---------|---------------------|
| 1º | **frequency** | 23,19% |
| 2º | recency_days | 12,19% |
| 3º | n_categories | 11,88% |
| 4º | avg_installments | 7,84% |
| 5º | avg_delivery_days | 7,41% |
| 6º | avg_review_score | 6,99% |
| 7º | monetary | 6,93% |
| 8º | top_category | 6,30% |
| 9º | avg_freight | 6,09% |
| 10º | avg_ticket | 6,08% |
| 11º | customer_state | 5,09% |

**Top 3, frequência, recência e diversidade de categorias, respondem por 47,3% da importância total**, e Top 5 por 62,5%.

**Achados-chave**:

- **`frequency` como principal driver** (23,19%): confirma a intuição RFM clássica, clientes que compram mais vezes têm menor propensão ao churn
- **`n_categories` em 3º lugar** (11,88%), muito próxima de `recency_days` (12,19%, diferença de apenas 0,3 ponto percentual): sugere que a diversidade de categorias é um forte indicador de engajamento, achado pouco documentado nos estudos anteriores sobre o Olist, incluindo Matuszelański e Kopczewska (2022). A Seção de estabilidade, mais adiante, confirma que esse trio se mantém robusto em 30 partições diferentes
- **`avg_review_score` apenas em 6º lugar** (6,99%): confirma a observação da EDA de que o problema não é qualidade do serviço, e sim engajamento

---

## Seção 8.5, Análise Comparativa de Representatividade, PSI e KS-Test

### O que faz

Compara a distribuição das 1.128 observações Churn originais com o subconjunto efetivamente retido no treino após o undersampling (902 no treino antes do balanceamento, reduzidas a 120 pela razão 3:1), usando duas métricas complementares: **PSI** (Population Stability Index) e **KS-test** (Kolmogorov-Smirnov, duas amostras).

### Por que faz assim

Após o undersampling aleatório, surge uma questão metodológica que a maioria dos estudos aplicados sobre o Olist **ignora**: o subconjunto de Churn que foi para o treino representa fielmente a classe Churn completa, ou o sorteio produziu uma amostra distorcida? Se o undersampling tivesse selecionado uma subamostra enviesada (por exemplo, predominantemente clientes de um estado, ou de valor muito baixo), o modelo teria aprendido padrões que não generalizam. A validação por PSI e KS é a forma rigorosa de descartar essa hipótese, e constitui uma das contribuições metodológicas deste trabalho.

**Critérios de interpretação**:

| Métrica | Faixa | Significado |
|---------|-------|-------------|
| PSI < 0,10 | Distribuições similares | Subconjunto representa bem a população |
| PSI 0,10 a 0,20 | Atenção | Algum deslocamento; investigar |
| PSI > 0,20 | Distorção significativa | Subconjunto não representa a população |
| KS p > 0,05 | Não rejeitar H0 | Distribuições estatisticamente equivalentes |
| KS p <= 0,05 | Rejeitar H0 | Distribuições significativamente diferentes |

### Como faz

```python
from scipy.stats import ks_2samp
for feature in features_numericas:
    psi = calcular_psi(churn_completo[feature], churn_treino[feature])
    ks_stat, p_valor = ks_2samp(churn_completo[feature], churn_treino[feature])
```

A função de PSI trata variáveis discretas de baixa cardinalidade (como `frequency`, com poucos valores únicos) usando contagem por valor único em vez de quantis, evitando faixas vazias que distorceriam o índice.

### Resultado obtido

Churn completo: 1.128 clientes. Churn no treino, antes do balanceamento: 902. Churn retido após o undersampling: 120 (taxa de retenção 10,6%).

| Feature | PSI | KS stat | p-valor | Status |
|---|---|---|---|---|
| recency_days | 0,0375 | 0,0574 | 0,8453 | Representativa |
| frequency | 0,0039 | 0,0041 | 1,0000 | Representativa |
| monetary | 0,0209 | 0,0418 | 0,9871 | Representativa |
| avg_ticket | 0,0265 | 0,0473 | 0,9583 | Representativa |
| avg_delivery_days | 0,0340 | 0,0537 | 0,8955 | Representativa |
| avg_review_score | 0,0103 | 0,0305 | 0,9999 | Representativa |
| avg_installments | 0,1293 | 0,1035 | 0,1818 | Atenção |
| avg_freight | 0,0602 | 0,0812 | 0,4476 | Representativa |
| n_categories | 0,0022 | 0,0101 | 1,0000 | Representativa |

**8 de 9 features representativas, 1 em atenção (`avg_installments`), nenhuma distorcida.** Mesmo `avg_installments`, a única fora da faixa "similar", tem p-valor de 0,1818 no KS-test, isto é, sem rejeição estatística da hipótese de mesma distribuição.

**Conclusão**: o undersampling aleatório com semente fixa preservou a distribuição estatística da classe Churn. O modelo aprendeu padrões da classe majoritária real, não de uma versão enviesada.

---

## Seção 8.6, Análise SHAP, Interpretabilidade Avançada

### O que faz

Calcula os **valores SHAP** (SHapley Additive exPlanations) via `TreeExplainer` para o modelo Random Forest, e produz quatro análises: beeswarm com ranking, SHAP médio por classe real, dependence plots das principais features e waterfall plots de clientes típicos.

### Por que faz assim

A importância via Gini (Seção 8) responde quais features importam. SHAP responde em que direção cada feature influencia as predições e em quais limiares, dimensão essencial para tradução em recomendações de negócio. Os valores SHAP, fundamentados na teoria dos jogos cooperativos de Shapley (1953), são o método de referência atual para explicabilidade local e global (LUNDBERG; LEE, 2017), e o TreeSHAP (LUNDBERG et al., 2020) calcula valores exatos para modelos de árvore, sem amostragem aproximada.

### Como faz

```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_te_sc)   # 236 clientes do hold-out x 11 features
```

### Resultado obtido

Valor base **E[f(x)] = 0,5042**: sem informação de features, o modelo prediz P(Churn) = 50,4% em média, refletindo o balanceamento 50/50 do treino.

**Ranking SHAP comparado ao Gini** (top 5):

| Rank SHAP | Feature | \|SHAP\| médio | Rank Gini | Δ Rank |
|---|---|---|---|---|
| 1 | frequency | 0,1144 | 1 | 0 |
| 2 | recency_days | 0,0668 | 2 | 0 |
| 3 | n_categories | 0,0636 | 3 | 0 |
| 4 | avg_installments | 0,0562 | 4 | 0 |
| 5 | top_category (enc.) | 0,0251 | 8 | +3 |

As quatro features mais relevantes ocupam posições **idênticas** nos dois rankings. A exceção informativa é `top_category`, que sobe da 8ª posição no Gini para a 5ª no SHAP, coerente com o viés conhecido do Gini contra variáveis categóricas.

**Direção dos efeitos** (correlação entre valor da feature e contribuição SHAP):

| Feature | Correlação | Leitura |
|---|---|---|
| `frequency` | -0,862 | Mais pedidos empurra a predição para Ativo (efeito protetor) |
| `recency_days` | +0,895 | Mais dias sem comprar empurra a predição para Churn |
| `n_categories` | -0,648 | Mais categorias empurra a predição para Ativo (efeito protetor) |

Os dependence plots revelam limiares não lineares: para `frequency`, o efeito é pronunciado entre 2 e 4 pedidos; para `recency_days`, há aceleração do risco acima de determinado patamar de dias.

> **Ressalva metodológica importante (revisão 2026-08).** A concordância entre Gini e SHAP **não constitui validação independente**: a importância de Gini mede redução de impureza nas árvores já ajustadas, e o TreeSHAP percorre exatamente essas mesmas árvores. Os dois métodos leem a mesma estrutura, então a concordância é esperada e não deve ser reportada como confirmação cruzada. Esse é o motivo de existir a análise de estabilidade a seguir, que usa uma fonte de evidência genuinamente independente (particionamento repetido dos dados e importância por permutação, que mede impacto preditivo em dados não vistos).

---

## Estabilidade da Importância entre Partições e Comparação entre Estimadores

*(numerada 4.6.1 no texto do TCC; sucede a análise SHAP no notebook)*

### O que faz

Substitui a alegação de "validação cruzada entre Gini e SHAP" por duas evidências efetivamente independentes: (i) a estabilidade da ordenação por Gini em **30 partições estratificadas independentes** do conjunto de treino; (ii) a comparação com a **importância por permutação**, o único dos três estimadores que mede o impacto real sobre dados não vistos, em vez de ler a estrutura interna das árvores.

### Por que faz assim

Gini e SHAP, como visto na seção anterior, compartilham a mesma fonte de informação (a estrutura das árvores ajustadas): concordarem entre si não prova nada de novo. Repetir o treino em partições diferentes dos dados, e comparar contra um estimador que usa embaralhamento de features em dados de validação, são testes de robustez de fato independentes.

### Como faz

Treina o Random Forest em 30 amostras estratificadas distintas e registra a posição de cada feature no ranking de Gini em cada uma. Em paralelo, calcula a importância por permutação (embaralhando cada feature e medindo a queda de desempenho) sobre o modelo de produção.

### Resultado obtido

**Estabilidade posicional (30 partições):**

| Feature | Posição média | Mín | Máx | Vezes no Top 3 (de 30) |
|---|---|---|---|---|
| frequency | 1,00 | 1 | 1 | 30 |
| n_categories | 2,30 | 2 | 4 | 29 |
| recency_days | 3,17 | 2 | 5 | 22 |
| monetary | 5,00 | 3 | 8 | 4 |
| avg_delivery_days | 6,30 | 3 | 10 | 2 |
| avg_review_score | 6,47 | 3 | 11 | 2 |

`frequency` ocupa o 1º lugar em todas as 30 partições. `n_categories` fica no Top 3 em 29 de 30. `recency_days` fica no Top 3 em 22 de 30, oscilando entre a 2ª e a 5ª posição. **É esta a evidência de estabilidade citada no TCC**, e ela confirma que o trio dominante (frequência, diversidade de categorias, recência) não é artefato de uma única partição.

**Gini versus importância por permutação:** a divergência mais notável é `avg_installments`, que ocupa a 4ª posição por Gini mas a **1ª por permutação** (média 0,0788, acima até de `frequency`). Isso indica que o número de parcelas tem impacto preditivo real sobre dados não vistos maior do que sua contribuição para a redução de impureza durante o treino sugere. Já `n_categories` cai da 3ª posição por Gini para a 7ª por permutação, e `top_category` e `customer_state` chegam a apresentar importância por permutação **negativa**, sinal de que embaralhar essas variáveis não piora (e às vezes melhora ligeiramente) o desempenho fora da amostra.

**Leitura conjunta**: `frequency` e `recency_days` são robustas pelos dois critérios. `n_categories` é robusta na estabilidade posicional (Gini), mas seu impacto preditivo isolado por permutação é mais modesto, o que é coerente com a natureza correlacional (não causal) desse achado, já ressalvada na Seção 1.1 e na Seção 5.

---

## Seção 9, Score de Churn e Segmentação de Risco

### O que faz

Aplica o Random Forest treinado a toda a base de 1.178 clientes, atribuindo a cada um uma **probabilidade contínua** P(Churn), e segmenta os clientes em quatro faixas de risco com base em cortes fixos (0,25 / 0,50 / 0,75).

### Por que faz assim

**Reposicionamento metodológico**: a análise de métricas robustas (Seção 7) mostrou que a classificação binária tem qualidade limitada neste dataset (MCC em torno de 0,19), consequência direta do tamanho reduzido da classe Ativo. Por isso, o entregável principal não é uma decisão sim/não, e sim uma probabilidade contínua que ordena os clientes por risco. Essa mudança preserva toda a informação da predição e é mais útil operacionalmente: em vez de forçar uma decisão binária com support de 10 amostras, a segmentação em faixas permite priorizar investimento até o limite do orçamento de retenção, e o custo de aquisição pode ser até 5 vezes o de retenção (REICHHELD; SCHEFTER, 2000).

**Por que quatro faixas em quartis?** Equilibra granularidade operacional e simplicidade de gestão, alinhado a Kotler e Keller (2016), que recomendam segmentação por risco em campanhas de retenção.

### Como faz

```python
rfm['churn_proba']   = rf_model.predict_proba(X_all)[:, 1]
rfm['risk_segment']  = pd.cut(rfm['churn_proba'], bins=[0, 0.25, 0.50, 0.75, 1.001])
```

`pd.cut` é fechado à direita: os intervalos reais são `(0; 0,25]`, `(0,25; 0,50]`, `(0,50; 0,75]` e `(0,75; 1]`. Portanto Crítico é `P > 75%` e Baixo é `P <= 25%`.

### Resultado obtido, perfil por segmento

| Segmento | Clientes | % da base | Recência média (dias) | Monetário médio (R$) | Review médio | P(Churn) média |
|----------|----------|---|---|---|---|---|
| 🟢 Baixo | 27 | 2,3% | 72,52 | 418,49 | 4,17 | 0,15 |
| 🟡 Médio | 101 | 8,6% | 80,72 | 252,77 | 4,35 | 0,40 |
| 🟠 Alto | 365 | 31,0% | 95,78 | 307,11 | 4,08 | 0,65 |
| 🔴 Crítico | 685 | 58,1% | 147,95 | 302,33 | 4,26 | 0,85 |

### Validação da segmentação pela taxa de churn observada

A tabela acima mostra apenas quantos clientes caem em cada faixa; isso não demonstra que as faixas separam risco de fato. A prova está no rótulo `churn` real, externo ao score: se a taxa de churn observada cresce de forma monotônica do Baixo ao Crítico, a segmentação captura risco genuíno.

| Faixa | n | Churn | Ativo | Taxa de churn | Lift sobre Ativo |
|---|---|---|---|---|---|
| Baixo | 27 | 6 | 21 | 22,2% | 18,32x |
| Médio | 101 | 80 | 21 | 79,2% | 4,90x |
| Alto | 365 | 358 | 7 | 98,1% | 0,45x |
| Crítico | 685 | 684 | 1 | 99,9% | 0,03x |

Monotonicidade estrita confirmada (Baixo < Médio < Alto < Crítico), com amplitude de **77,6 pontos percentuais** entre a faixa Baixo e a faixa Crítico. Como a taxa base de churn já é 95,8%, o lift sobre Churn tem teto natural em torno de 1,04x; a leitura relevante está no lift sobre **Ativo**: a faixa Baixo concentra 21 dos 50 clientes Ativos (42,0%) em apenas 2,3% da base (lift 18,3x), e Baixo + Médio juntas concentram 42 dos 50 Ativos (84,0%) em 10,9% da base (lift 7,73x).

**Atenção à composição da base completa**: os 40 clientes Ativos usados no ajuste do modelo caem todos nas faixas Baixo e Médio, o que infla esse recorte quando calculado sobre a base completa (a base completa inclui as linhas vistas no treino). O recorte que sobrevive fora da amostra, isolado no diagnóstico a seguir, é a exclusão da faixa Crítico: no hold-out, 9 dos 10 Ativos (90,0%) ficam fora do Crítico, em 39,0% da base, lift de 2,31x. Na base completa, o mesmo recorte dá 49 de 50 Ativos (98,0%) em 41,9% da base, lift de 2,34x. **É este o número (não o de Baixo + Médio) reportado nas Seções 4.2 e 4.7 do TCC como validação principal.**

### Diagnósticos comparativos, por que o Random Forest e não o HistGradient Boosting

O HistGradient Boosting vence o Random Forest em MCC, Kappa, F1-Macro, Precisão-Ativo e Recall-Churn (ver leaderboard da Seção 7). A escolha do Random Forest para produção precisa ser sustentada com um argumento adicional, e é isso que os diagnósticos a seguir fazem, comparando a **dispersão dos scores** dos dois modelos (mais a Regressão Logística, como referência linear) na base completa.

**Dispersão dos scores (n = 1.178):**

| Modelo | Mín | P25 | Mediana | P75 | Máx | Limiar |
|---|---|---|---|---|---|---|
| Random Forest | 0,0033 | 0,6545 | 0,7781 | 0,8601 | 0,9934 | 0,38 |
| HistGradient Boosting | 0,0006 | 0,7927 | 0,9621 | 0,9915 | 0,9999 | 0,18 |
| Logistic Regression | 0,0009 | 0,4204 | 0,6000 | 0,7818 | 0,9988 | 0,12 |

**Ocupação das quatro faixas de risco, com os mesmos cortes da Tabela 7:**

| Modelo | Baixo | Médio | Alto | Crítico | Maior faixa |
|---|---|---|---|---|---|
| Random Forest | 2,3% | 8,6% | 31,0% | 58,1% | 58,1% |
| HistGradient Boosting | 9,6% | 5,6% | 7,2% | 77,6% | 77,6% |
| Logistic Regression | 5,8% | 31,2% | 33,4% | 29,6% | 33,4% |

O HistGradient Boosting **satura os scores junto do extremo superior** da escala (mediana 0,9621), esvaziando as faixas intermediárias e concentrando 77,6% da base em Crítico. O limiar ótimo baixo do HistGB (0,18) não indica probabilidades comprimidas perto de zero: é o oposto, como praticamente todo score excede esse valor, a varredura que maximiza Kappa converge para um corte baixo simplesmente porque a maior parte da base já está acima dele. Esse deslocamento das probabilidades para a classe majoritária do treino é o efeito descrito por Dal Pozzolo et al. (2015): preserva a **ordenação** relativa dos clientes (por isso o AUC do HistGB não colapsa), mas inviabiliza uma segmentação em quatro faixas operacionalmente útil. É esse o argumento numérico que sustenta a escolha do Random Forest como modelo de produção, apesar de não vencer em todas as métricas de classificação.

**Concentração de Ativos nas faixas de menor risco** (Baixo + Médio, base completa): Random Forest 42/50 (84,0%) em 10,9% da base, lift 7,73x; HistGradient Boosting 43/50 (86,0%) em 15,2%, lift 5,66x; Logistic Regression 32/50 (64,0%) em 37,0%, lift 1,73x.

### Diagnóstico restrito ao hold-out

Repete a mesma validação apenas sobre os 236 clientes nunca vistos pelo modelo, separando também os 50 Ativos por origem: 10 no hold-out, 40 vistos no ajuste (retidos pelo undersampling) e 0 descartados (todo Ativo é preservado; apenas Churn é reduzido pelo undersampling).

**Faixas no hold-out, Random Forest (n = 236, Ativos = 10):**

| Faixa | n | Churn | Ativo | Taxa de churn | Lift Ativo |
|---|---|---|---|---|---|
| Baixo | 4 | 3 | 1 | 75,0% | 5,90x |
| Médio | 15 | 14 | 1 | 93,3% | 1,57x |
| Alto | 73 | 66 | 7 | 90,4% | 2,26x |
| Crítico | 144 | 143 | 1 | 99,3% | 0,16x |

A progressão se mantém no hold-out (75,0% no Baixo, 99,3% no Crítico), com uma inversão entre Médio e Alto atribuível ao número reduzido de observações por faixa nesse subconjunto (15 e 73 clientes), não a uma quebra real de monotonicidade. A leitura principal continua sendo a da base completa, que é o escopo da Tabela 7 do TCC.

**Comparação do recorte "fora da faixa Crítico" entre base completa e hold-out**, o número efetivamente citado nas Seções 4.2 e 4.7:

| Modelo | Hold-out (Ativos / cobertura / lift) | Base completa (Ativos / cobertura / lift) |
|---|---|---|
| Random Forest | 9/10, 39,0%, 2,31x | 49/50, 41,9%, 2,34x |
| HistGradient Boosting | 3/10, 20,3%, 1,48x | 43/50, 22,4%, 3,84x |
| Logistic Regression | 10/10, 66,9%, 1,49x | 46/50, 70,4%, 1,31x |

O lift do Random Forest é consistente entre base completa e hold-out (2,34x contra 2,31x), o que reforça que a segmentação generaliza para dados nunca vistos. O HistGradient Boosting, em contraste, tem lift de base completa muito mais alto (3,84x) do que no hold-out (1,48x), sinal de que parte de sua vantagem aparente vem de memorização.

---

## Seção 10, Exportar Resultados

### O que faz

Grava um CSV com, para cada cliente: identificador, features RFM resumidas, classe real, probabilidade predita e segmento de risco.

### Por que faz assim

O CSV é a interface entre o trabalho científico e a operação: permite auditoria das predições, integração com ferramentas de marketing e anexação ao TCC como artefato de evidência, sem exigir que o analista de CRM rode o notebook.

### Como faz

```python
rfm[['customer_unique_id', 'recency_days', 'frequency', 'monetary',
     'avg_review_score', 'top_category', 'customer_state',
     'churn', 'churn_proba', 'risk_segment']].to_csv('olist_churn_predictions.csv')
```

### Resultado obtido

Arquivo `olist_churn_predictions.csv` com **1.178 linhas e 10 colunas**, salvo em `notebook/`.

---

## Figuras de Fechamento (Figura 1 e Figura 2 do TCC)

Duas células finais desenham os diagramas de síntese do TCC, com **todos os rótulos derivados do estado real do notebook** (número de tabelas, de features, de algoritmos, de folds, datas efetivas das janelas), em vez de valores escritos à mão. Essa decisão corrige um problema real de uma versão anterior, em que a figura afirmava "5 tabelas" e "jan a jun de 2018" enquanto o código já usava seis tabelas e uma janela de predição de cinco meses; com valores derivados, essa divergência se torna estruturalmente impossível.

**Figura 1, pipeline metodológico**: tabelas relacionais = 6; observação = set/2016 a dez/2017 (16 meses); predição = jan a mai/2018 (5 meses); features = 11; filtro = `frequency >= 2`; algoritmos = 10; folds da validação cruzada = 5; limite de referência do gap CV-hold-out = 0,15.

**Figura 2, janela temporal dupla**: o início da observação vem do primeiro pedido real da base (`obs_orders['order_purchase_timestamp'].min()`), não da constante `OBS_START`; as larguras das barras são proporcionais à duração efetiva de cada janela. Pedidos na janela de observação: 43.695. Pedidos na janela de predição: 34.174.

Ambas as figuras são salvas em `notebook/` (`figura1_pipeline.png`, `figura2_janelas.png`) apenas quando as células correspondentes são executadas; não ficam versionadas prontas no repositório.

---

## Síntese Final

### Os principais achados em uma página

**Sobre o modelo**:
- Random Forest selecionado para produção, ROC-AUC = 0,8106 no hold-out
- Não é o melhor modelo em todas as métricas (Regressão Logística lidera MCC e Kappa, HistGradient Boosting lidera G-Mean), mas é o único cuja distribuição de scores viabiliza a segmentação em quatro faixas operacionalmente úteis
- Recall-Churn de 97,4%, captura a grande maioria dos clientes em risco
- Nenhum modelo foi descartado por overfitting; o Δ CV-holdout é reportado, não usado como filtro

**Sobre os drivers de churn** (Gini + permutação + estabilidade entre partições):
- **Frequência de compras** é o driver mais robusto: 1º lugar por Gini (23,2%) e 1º em 30 de 30 partições
- **Diversidade de categorias** (`n_categories`) é estável posicionalmente (Top 3 em 29 de 30 partições), mas seu impacto isolado por permutação é mais modesto, coerente com a leitura correlacional (não causal) já ressalvada
- **Recência** completa o trio dominante, no Top 3 em 22 de 30 partições
- **Satisfação declarada** (`avg_review_score`) tem importância secundária: o churn no Olist é, sobretudo, um problema de engajamento, não de qualidade

**Sobre a metodologia**:
- O filtro `frequency >= 2` é empiricamente validado: melhora o AUC em 0,2875 ponto
- O undersampling aleatório com semente fixa é estatisticamente representativo (8 de 9 features com PSI < 0,10, nenhuma distorcida)
- A divergência entre rankings de AUC e métricas robustas (Spearman entre 0,30 e 0,65) confirma a necessidade do portfólio multi-métrica
- A concordância entre Gini e SHAP não é tratada como validação independente; a evidência de robustez vem da estabilidade entre partições e da comparação com importância por permutação
- O protocolo de validação cruzada e de ajuste de limiar foi corrigido para eliminar dois vazamentos de informação identificados em revisão

**Sobre a aplicação de negócio**:
- 1.178 clientes com score e segmento de risco atribuídos
- A faixa Crítico concentra 58,1% da base; a exclusão dela isola 9 dos 10 Ativos do hold-out (90,0%) em 39,0% da base, número consistente com a base completa (2,31x contra 2,34x de lift)
- A diversidade de categorias como driver sugere cross-selling como mecanismo promissor de retenção, com a ressalva de que a evidência é correlacional

### Limitações reconhecidas

1. **Base reduzida**: 1.178 clientes após o filtro, com apenas 50 Ativos
2. **Período histórico**: o dataset cobre 2016 a 2018, e pode não refletir dinâmicas atuais do mercado
3. **MCC e Kappa moderados** (em torno de 0,19): consequência matemática do hold-out com apenas 10 Ativos, não falha algorítmica (LUQUE et al., 2019)
4. **Variabilidade entre versões de bibliotecas**: mesmo com semente fixa, mudanças de implementação no scikit-learn podem produzir variações marginais nas métricas e na ordenação de features próximas

### Trabalhos futuros sugeridos

- Hiperparametrização sistemática (Optuna)
- SHAP por instância para explicações personalizadas de retenção
- Modelos de sobrevivência (Cox) para estimar tempo até o churn
- Deep Learning sequencial (LSTM) para padrões temporais em sequências de compra
- Features de sessão e navegação para aumentar o poder preditivo
- Deploy via API com monitoramento de drift

---

## Glossário rápido

| Termo | Definição |
|-------|-----------|
| **Churn** | Cliente que não retorna no período de predição |
| **RFM** | Recency, Frequency, Monetary, framework clássico de análise comportamental |
| **Data leakage** | Vazamento de informação futura (ou do teste) para o treino |
| **Out-of-fold (OOF)** | Probabilidades preditas para cada observação quando ela estava na porção de validação do seu fold, nunca vista no ajuste daquele fold |
| **Hold-out** | Conjunto separado para avaliação final, nunca visto durante treino ou ajuste de limiar |
| **AUC (ROC-AUC)** | Área sob a curva ROC, mede capacidade discriminativa independente do limiar |
| **F1-Macro** | Média harmônica de Precisão e Recall, calculada por classe e depois mediada |
| **MCC** | Matthews Correlation Coefficient, métrica simétrica para classificação binária |
| **G-Mean** | Raiz quadrada do produto entre Sensibilidade e Especificidade |
| **Kappa de Cohen** | Concordância corrigida pelo acerto esperado ao acaso |
| **Average Precision (AP)** | Área sob a curva Precision-Recall |
| **SHAP** | SHapley Additive exPlanations, explicabilidade baseada em teoria dos jogos |
| **Importância por permutação** | Queda de desempenho ao embaralhar uma feature nos dados de validação; mede impacto preditivo real, não estrutura interna do modelo |
| **PSI** | Population Stability Index, mede deslocamento entre distribuições |
| **KS-test** | Kolmogorov-Smirnov, teste de equivalência distribucional |
| **Threshold (limiar)** | Valor de probabilidade acima do qual o cliente é classificado como Churn |
| **Stratified split** | Divisão treino/teste que preserva a proporção das classes |

---

## Referências mobilizadas

- **HUGHES (1994)**, Strategic Database Marketing: metodologia RFM (Seção 3)
- **KAUFMAN et al. (2012)**, Leakage in data mining: janelas temporais e ajuste do scaler (Seções 2, 6)
- **MATUSZELAŃSKI; KOPCZEWSKA (2022)**, Estudo Olist: filtro de frequência (Seções 3, 4, 4.1)
- **HE; GARCIA (2009)**, Imbalanced data: balanceamento e representatividade (Seções 6, 8.5)
- **CHAWLA et al. (2002)**, SMOTE: comparação de estratégias (Seção 6)
- **HADDADI et al. (2024)**, Resampling para churn: estratégia adotada (Seção 6)
- **BREIMAN (2001)**, Random Forest: modelo de produção e importância de Gini (Seções 7, 8)
- **FRIEDMAN (2001)**, Gradient Boosting: algoritmos comparados (Seção 7)
- **VAFEIADIS et al. (2015)**, Comparação de algoritmos para churn: desenho do benchmark (Seção 7)
- **FAWCETT (2006)**, ROC analysis: ajuste de limiar (Seção 7)
- **DE LA CRUZ HUAYANAY; BAZÁN; RUSSO (2025)**, Métricas para classes desbalanceadas: protocolo multi-métrica e critério do limiar (Seções 7, 7.1)
- **DAL POZZOLO et al. (2015)**, Calibração sob undersampling: deslocamento das probabilidades (Seções 7, 9)
- **LUQUE et al. (2019)**, Impacto do desbalanceamento em métricas: interpretação de MCC e Kappa moderados (Seção 7)
- **KUBAT; MATWIN (1997)**, One-sided selection: discussão metodológica (Seção 8.5)
- **SHAPLEY (1953)**, Teoria dos jogos: fundamento do SHAP (Seção 8.6)
- **LUNDBERG; LEE (2017)**, Unified approach to interpreting: SHAP (Seção 8.6)
- **LUNDBERG et al. (2020)**, TreeSHAP: eficiência computacional (Seção 8.6)
- **MAAN; MAAN (2023)**, Explainable ML para churn: SHAP por instância (Seção 8.6)
- **EL ATTAR; EL-HAJJ (2026)**, Explainable AI-driven churn prediction: contexto (Seção 8.6)
- **REICHHELD; SCHEFTER (2000)**, E-loyalty: fundamento econômico da segmentação (Seção 9)
- **KOTLER; KELLER (2016)**, Administração de Marketing: segmentação por risco (Seção 9)
- **HADDEN et al. (2007)**, Churn management: definição de churn implícito (Seção 4)
- **NESLIN et al. (2006)**, Defection detection: definição de churn implícito (Seção 4)
- **IMANI et al. (2025)**, Revisão sistemática: riscos metodológicos e trabalhos futuros (Seções 2, Síntese)
- **MANZOOR et al. (2024)**, Recomendações para praticantes: features complementares ao RFM (Seção 3)
- **NICULESCU-MIZIL; CARUANA (2005)**, Probabilidades bem calibradas: contexto de calibração (Seção 9)
- **OLIST (2018)**, Dataset Olist no Kaggle: fonte de dados (Seção 1)
