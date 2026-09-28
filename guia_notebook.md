# Guia Completo do Notebook `olist_churn_prediction.ipynb`

**Análise Preditiva de Churn em E-Commerce Brasileiro**
TCC, MBA em Inteligência Artificial e Big Data, ICMC/USP
Autor: Fabiano Rodrigues Ugulino. Orientadora: Profa. Dra. Cibele M. Russo

---

## Sobre este guia

Este documento explica, em ordem sequencial, o que o notebook faz, por que faz e o que ele encontra. O objetivo é que qualquer pessoa, mesmo sem familiaridade com Machine Learning aplicado a dados de e-commerce, consiga acompanhar o raciocínio do início ao fim, entender as decisões metodológicas e reproduzir o trabalho.

A estrutura segue exatamente a do notebook: Seções 0, 1, 1.1, 2, 3, 4, 4.1, 5, 6, 7, 7.1, 8, 8.5, 8.6, a comparação SHAP entre os três modelos finalistas, a análise de estabilidade da importância, 9 (incluindo os diagnósticos comparativos e a curva de captura em cobertura fixa), 10, as duas figuras de fechamento e o sumário final. Cada bloco é apresentado em quatro partes: **O que faz**, **Por que faz assim** (justificativa metodológica), **Como faz** (descrição técnica) e **Resultado obtido** (números da execução de referência, seed 42).

> **Revisão de setembro de 2026.** Este guia foi atualizado para refletir a versão final do notebook e a monografia entregue (`docs/tcc.pdf`, declaração datada de 21/09/2026). Em relação à revisão de agosto, entram a escolha declarada do modelo de produção, a comparação SHAP entre Random Forest, HistGradient Boosting e Regressão Logística (Tabela 7 do TCC), a curva de captura em cobertura fixa, a célula de consolidação numérica e a renumeração de tabelas, figuras e seções conforme o PDF final. A leitura da importância por permutação também foi corrigida: no regime amostral deste trabalho ela não é informativa, e não deve ser usada como evidência de impacto preditivo. Os números da execução de referência não mudaram.

---

## Correspondência entre notebook e TCC

| Notebook | TCC (PDF final) | Tabelas, quadros e figuras |
|---|---|---|
| Seções 1 e 1.1 | 1.2 e 3.1 | |
| Seção 2 | 3.2 | Figura 2 |
| Seção 3 | 3.3 | Tabela 1 |
| Seções 4 e 4.1 | 3.3 (validação do filtro) e 3.4 (target) | |
| Seção 5 | 4.1 | Figuras 3 a 6 |
| Seção 6 | 3.4 | Quadro 1, Tabelas 2 e 3 |
| Seção 7 | 3.5, 4.2 e 4.4 | Tabela 4, Quadro 2, Figuras 7 a 10 |
| Seção 7.1 | 4.3 | |
| Seção 8 | 4.6 | Tabela 6, Figuras 11 e 12 |
| Seção 8.5 | 4.5 e 4.5.1 | Tabela 5 |
| Seção 8.6 | 4.6 e 4.6.1 | Figuras 13 a 19 |
| SHAP nos três modelos | 4.6 | Tabela 7 |
| Estabilidade entre partições | 4.6 | |
| Seção 9 (segmentação e diagnósticos) | 4.2 e 4.7 | Tabela 8, Figura 20 |
| Curva de captura em cobertura fixa | 4.2, 4.7 e Conclusão | |
| Figuras de fechamento | 3 (abertura) e 3.2 | Figuras 1 e 2 |

---

## Visão geral da proposta

### O problema

A Olist é uma plataforma de marketplace que conecta pequenos varejistas brasileiros a grandes canais de venda. Como em todo e-commerce, uma parcela significativa dos clientes compra uma única vez e nunca mais retorna: fenômeno conhecido como **churn**. Antecipar quais clientes têm maior risco de abandono permite que a empresa direcione esforços de retenção (cupons, ofertas, contato) de forma proativa, em vez de reagir à perda já consumada.

### A pergunta de pesquisa

> Como identificar, de forma antecipada e com base apenas em dados transacionais históricos, quais clientes da Olist apresentam maior propensão a não voltar a comprar, e qual é o perfil do cliente que retorna?

### A abordagem em uma frase

Construir um modelo preditivo que, a partir de variáveis comportamentais de cada cliente (recência, frequência, valor, satisfação, padrão de pagamento, diversidade de categorias), estime a probabilidade de churn nos meses seguintes e ordene os clientes por risco, com rigor metodológico para evitar vazamento de informação e validar a generalização.

### O que torna este trabalho diferente

- **Janela temporal dupla**, com observação e predição estritamente separadas, eliminando *data leakage*
- **Validação empírica do filtro de compradores únicos** (Seção 4.1): não é argumento puramente teórico, é demonstrado por ablação
- **Protocolo multi-métrica** (Seções 7 e 7.1): avalia AUC, Average Precision, F1-Macro, MCC, G-Mean e Kappa em conjunto, e mede a concordância entre rankings pela correlação de Spearman, respondendo a uma limitação metodológica documentada por De la Cruz Huayanay, Bazán e Russo (2025)
- **Reamostragem e limiar sem vazamento**: o balanceamento ocorre dentro de cada fold da validação cruzada, e o limiar de decisão é derivado das probabilidades *out-of-fold* do treino, nunca do conjunto de teste. O efeito da correção é quantificado (queda de até 0,3251 na CV-AUC)
- **Escolha declarada do modelo de produção**, fundamentada na qualidade da ordenação e verificada por **captura em cobertura fixa**, em vez de eleger automaticamente o maior valor de uma métrica
- **Validação de representatividade do balanceamento** (Seção 8.5, PSI e KS-test)
- **Estabilidade da importância entre partições** e comparação SHAP entre três modelos, em vez de tratar a concordância Gini/SHAP como validação independente (os dois métodos leem a mesma estrutura de árvores)

### Bibliotecas usadas

```python
pandas, numpy           # manipulação e estatística
scikit-learn             # os dez algoritmos, métricas, pré-processamento, permutação
imbalanced-learn         # SMOTE, usado apenas na comparação de estratégias (Seção 6)
matplotlib, seaborn      # visualização
shap                     # explicabilidade (Seção 8.6 e comparação entre modelos)
scipy                    # correlação de Spearman, PSI e KS-test
```

---

## Seção 0, Imports e Configurações Globais

### O que faz

Carrega todas as bibliotecas necessárias, define a paleta de cores dos gráficos e fixa as constantes que delimitam as janelas temporais do estudo.

### Por que faz assim

Concentrar imports e constantes numa única célula garante que qualquer alteração de janela temporal se propague por todo o notebook a partir de um só ponto, evitando o erro comum de datas repetidas em várias células que saem de sincronia.

A semente fixa (`random_state = 42`) é aplicada em todas as operações estocásticas: divisão treino/teste, balanceamento, modelos com randomização interna. Isso garante que executar o notebook duas vezes na mesma máquina produza os mesmos números. As únicas exceções deliberadas são as análises de robustez em trinta partições, que variam a semente de particionamento de 0 a 29.

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

O dataset Olist é a maior base pública de e-commerce brasileiro disponível, com cerca de 100 mil pedidos reais entre 2016 e 2018. Sua estrutura relacional (seis tabelas conectadas por chaves) reflete a realidade de qualquer e-commerce corporativo. O notebook inclui um diagrama entidade-relacionamento com as cardinalidades: 1,13 itens por pedido, 1,04 pagamentos por pedido e 99,8% dos pedidos com avaliação.

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

**Split temporal**: este é o elemento metodológico mais importante do pipeline. Sem janelas separadas, é fácil construir um modelo que "vê o futuro", por exemplo calculando a recência com dados do mesmo período em que se está predizendo o churn. Esse fenômeno, conhecido como *data leakage* (KAUFMAN et al., 2012), produz métricas artificialmente altas no treino que desabam em produção. A literatura recente identifica este como um dos principais riscos metodológicos em estudos de churn (IMANI et al., 2025).

**Horizonte de cinco meses**: o TCC (Seção 3.2) justifica a escolha como equilíbrio entre sensibilidade à recompra e representatividade do horizonte preditivo, compatível com um e-commerce dominado por clientes de compra única (MATUSZELAŃSKI; KOPCZEWSKA, 2022).

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

Une pedidos, clientes, itens, produtos, pagamentos e avaliações da janela de observação, calcula o tempo de entrega em dias, e agrega tudo por `customer_unique_id`, produzindo **onze features** por cliente (Tabela 1 do TCC).

### Por que faz assim

A metodologia **RFM (Recência, Frequência, Valor Monetário)** é um dos frameworks mais consolidados de análise comportamental de clientes, com origem no marketing direto (HUGHES, 1994). As três dimensões capturam de forma objetiva o que importa em um relacionamento comercial: quando o cliente comprou pela última vez, com que frequência ele compra e quanto ele gasta.

RFM puro é incompleto para predição de churn moderna. A literatura recente (MANZOOR et al., 2024) recomenda enriquecer com variáveis adicionais que capturem engajamento, satisfação e padrão de pagamento. Por isso, foram adicionadas oito features complementares: ticket médio, tempo médio de entrega, nota média das avaliações, número médio de parcelas, frete médio, número de categorias distintas, categoria principal e estado do cliente.

O ponto mais delicado do cálculo é `monetary`. O join intermediário replica o pagamento uma vez por item (43.695 pedidos viram 52.915 linhas): somar `payment_value` diretamente inflaria o gasto de quem comprou vários produtos no mesmo pedido. A solução reconstrói o valor de cada pedido a partir de `payments_df` e agrega por cliente contando cada pedido uma única vez, garantindo que `avg_ticket = monetary / frequency` seja de fato o ticket médio por pedido. Três `assert` verificam essa consistência ao final da célula.

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

Nulos são tratados por critério específico de cada variável, não por um valor único: mediana para avaliações e pagamentos, zero para frete (ausência significa frete grátis), `1` para parcelas (à vista) e o rótulo `desconhecido` para categoria ausente. Esse tratamento ocorre inteiramente na janela de observação, antes da definição do target, e portanto sem acesso a informação da janela de predição.

> **Duas ressalvas registradas no TCC.** (i) O rótulo `desconhecido` é contado como categoria distinta por `COUNT(DISTINCT ...)`, de modo que um cliente com pedido nessa condição pode ter `n_categories` acrescido em uma unidade (nota da Tabela 1). (ii) O Label Encoding de `top_category` e `customer_state` faz os modelos de árvore tratarem o inteiro atribuído a cada categoria como grandeza ordenada, o que recomenda cautela na leitura da posição de `top_category` nos rankings de importância. Codificações alternativas (target encoding, agrupamento de categorias raras) constam entre os trabalhos futuros.

### Resultado obtido

Join completo: 52.915 linhas (uma por item de pedido), 20 colunas. Categorias conhecidas em 51.972 de 52.915 itens (98,2%); o restante (1,8%) fica marcado como `desconhecido`. Sem nulos remanescentes após a imputação.

Agregando por cliente, o resultado é a base **ainda sem o filtro de recorrência**: **42.395 clientes únicos**, com onze features cada. Esta é a base `rfm_unfiltered`, reutilizada na validação empírica da Seção 4.1.

---

## Seção 4, Definição do Target, Churn por Janela de Tempo

### O que faz

Cria a variável-alvo binária (`churn = 1` para clientes que **não** compraram na janela de predição, `churn = 0` para os que compraram pelo menos uma vez) e aplica o filtro `frequency >= 2`, restringindo a base a clientes com histórico de recompra.

### Por que faz assim

A definição comportamental de churn, em oposição à contratual (cancelamento explícito), é a única aplicável a e-commerce, onde não existe contrato a ser rescindido: o abandono é inferido pela ausência de nova compra em um horizonte predefinido (NESLIN et al., 2006; HADDEN et al., 2007).

**Por que filtrar `frequency >= 2`?** Compradores únicos seriam classificados como Churn de forma trivial, sem que o modelo aprendesse padrão comportamental real, e não têm histórico suficiente para caracterizar recompra. Matuszelański e Kopczewska (2022), no estudo mais completo disponível sobre o Olist, recomendam esse filtro. A base `rfm_unfiltered` (42.395 clientes) é preservada justamente para permitir a validação empírica que se segue na Seção 4.1, em vez de aceitar a decisão apenas por autoridade.

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

O **desbalanceamento extremo** (~96/4) é a característica que mais define os desafios técnicos do problema e condiciona todo o restante do pipeline. O TCC (Seção 3.4) observa que essa proporção ocorre numa coorte composta apenas por clientes recorrentes: mesmo entre quem já recomprou, os intervalos entre transações tendem a exceder o horizonte de cinco meses.

---

## Seção 4.1, Validação Empírica da Exclusão de Compradores Únicos

### O que faz

Roda um experimento de ablação: treina o mesmo Random Forest de referência, com o mesmo protocolo de avaliação, em duas configurações (com e sem o filtro `frequency >= 2`), e compara AUC e MCC.

### Por que faz assim

A exclusão de compradores únicos pode parecer intuitiva, mas decisões metodológicas de peso não devem ser aceitas apenas por intuição num trabalho científico. A questão concreta é: incluir 41 mil clientes a mais ajudaria o modelo, ou apenas adicionaria ruído? Esta seção responde com evidência empírica, em vez de apenas justificar em texto. É também a resposta direta a um questionamento recorrente em bancas: "por que descartar cerca de 96% dos clientes?"

Nesta célula também são definidas as funções e constantes centrais reaproveitadas em todo o notebook:

| Elemento | Papel |
|---|---|
| `balance_dataset` | Undersampling da maioria até `RATIO_MAJ = 3` vezes a minoria, mais oversampling da minoria com ruído gaussiano |
| `probabilidades_out_of_fold` | Validação cruzada com imputação, escala e reamostragem ajustadas dentro de cada fold |
| `limiar_out_of_fold` | Varredura do limiar sobre as probabilidades out-of-fold do treino |
| `CV_SPLITTER` | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| `THRESHOLD_SCAN` | 200 valores entre 0,01 e 0,99 |
| `CRITERIO_LIMIAR` | `'kappa'` (adotado); `'f1'` como sensibilidade |
| `GAP_MAX` | 0,15, referência para a verificação de `|CV - hold-out|`, reportada mas não usada para descartar modelos |

### Como faz

Mesmas features, mesmo modelo (Random Forest, `n_estimators=300`, `max_depth=10`), mesmo split estratificado 80/20, mesma semente. O limiar de decisão de cada configuração é obtido das probabilidades *out-of-fold* do próprio treino, nunca do conjunto de teste.

### Resultado obtido

| Config | Clientes | Churn % | Ativos no teste | AUC | MCC | CV-AUC | Limiar |
|--------|----------|---------|------------------|-----|-----|--------|--------|
| **Adotada** (freq >= 2) | 1.178 | 95,8% | 10 | **0,8106** | **0,1931** | 0,6814 | 0,38 |
| Alternativa (freq >= 1, inclui únicos) | 42.395 | 98,9% | 94 | 0,5231 | 0,0400 | 0,5817 | 0,31 |

Incluir compradores únicos derruba o AUC em **0,2875 ponto** (aproximando o modelo do acaso) e o MCC em **79%**. **A exclusão de compradores únicos está empiricamente validada**: longe de ser uma escolha de conveniência, é uma decisão que melhora mensuravelmente o poder discriminativo do modelo. Estes números aparecem na Seção 3.3 do TCC.

---

## Seção 5, Análise Exploratória dos Dados (EDA)

### O que faz

Calcula estatísticas descritivas por classe e produz as visualizações que caracterizam o perfil comportamental de clientes Ativos versus Churn. O painel 2x2 origina as **Figuras 3 a 6** do TCC (recência, frequência, review score e proporção de classes), discutidas na Seção 4.1.

### Por que faz assim

A EDA cumpre três funções: gerar hipóteses descritivas sobre quais variáveis discriminam bem as classes; verificar a sanidade dos dados; e fornecer material de comunicação para contextos de negócio. É importante registrar que a EDA é descritiva, não causal nem preditiva: diferenças observadas aqui nem sempre se traduzem em alto poder preditivo no modelo final, devido a correlações entre features e efeitos de interação.

### Como faz

Painel 2x2 (recência, proporção de classes, boxplot de frequência, review score), painel 2x3 do perfil multivariado (seis distribuições sobrepostas, com a média de cada grupo marcada sobre a série completa e o eixo limitado ao percentil 95) e um gráfico de taxa de churn por categoria de produto, este último exploratório e fora do corpo do TCC.

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
- **Valor monetário e diversidade de categorias** são as dimensões que mais separam as classes na análise univariada: R$ 372,25 contra R$ 299,12 e 1,80 contra 1,54 categorias. A diversidade de categorias é revisitada com evidência preditiva nas Seções 8, 8.6 e na análise de estabilidade

---

## Seção 6, Pré-processamento e Balanceamento de Classes

### O que faz

Executa quatro operações em sequência: (i) Label Encoding das variáveis categóricas; (ii) split estratificado treino/teste 80/20; (iii) imputação de nulos por mediana e padronização, ajustadas apenas no treino; (iv) balanceamento do treino por undersampling da classe majoritária combinado a oversampling sintético da minoritária com ruído gaussiano.

### Por que faz assim

**Label Encoding**: as categóricas (`top_category` com dezenas de níveis, `customer_state` com 27 UFs) precisam virar numéricas. One-hot encoding criaria dezenas de colunas adicionais, incompatível com apenas 1.178 amostras. Label Encoding preserva a dimensionalidade, com a limitação registrada na Seção 3.

**`StratifiedShuffleSplit` 80/20**: preserva a proporção original de classes em ambos os conjuntos. Em desbalanceamento extremo, um split aleatório simples poderia deixar o teste sem nenhum cliente Ativo, inviabilizando a avaliação.

**Ajuste do imputador e do scaler apenas no treino**: se a mediana e o desvio padrão fossem calculados sobre o dataset completo, informação do teste vazaria para o treino (KAUFMAN et al., 2012).

**Balanceamento combinado (undersampling + oversampling com ruído gaussiano)**: a função `balance_dataset` não é SMOTE. Não há interpolação entre vizinhos; cada instância sintética é uma cópia de uma observação real da classe Ativo somada a ruído `N(0; 0,05)` sobre as features já padronizadas, o que preserva a vizinhança original em vez de criar pontos em regiões não observadas do espaço de features. O TCC identifica o procedimento como a replicação ruidosa (*noisy replication*) de Lee (2000) e o apoia no resultado de Bishop (1995), segundo o qual treinar com ruído aditivo de baixa variância equivale a uma regularização de Tikhonov. A razão de undersampling é `3:1` (até 3 amostras Churn por amostra Ativo), meio-termo entre balanceamento estrito (que descartaria muita informação Churn) e proporção original (que ignoraria a minoria).

**Balanceamento aplicado apenas ao treino**: o teste preserva a distribuição original (95,8% Churn), para que a avaliação reflita as condições reais de operação.

### Como faz

```python
n_maj_keep = min(len(maj), RATIO_MAJ * len(min))     # undersampling, RATIO_MAJ = 3
n_synth    = max(0, n_maj_keep - len(min))
X_synth    = amostras_min[escolhidas] + N(0, 0.05)    # oversampling
```

A célula guarda também os índices retidos (`maj_keep_idx`, `min_keep_idx`), usados adiante na análise de representatividade e no diagnóstico fora da amostra.

### Resultado obtido

| Conjunto | Tamanho | Distribuição |
|----------|---------|--------------|
| Treino original | 942 | 40 Ativos / 902 Churn |
| Treino balanceado | 240 | 120 Ativos / 120 Churn |
| Teste (hold-out) | 236 | 10 Ativos / 226 Churn (95,8% Churn, original) |

Onze features ao todo: 9 numéricas e 2 categóricas codificadas. Dois terços da classe Ativo no treino balanceado são sintéticos (80 de 120).

### Comparação empírica das estratégias de balanceamento

Avalia quatro alternativas (sem balanceamento, undersampling puro 1:1, SMOTE e a combinação adotada under 3:1 + ruído gaussiano), primeiro num split de referência e depois em **trinta divisões independentes**, com reamostragem sempre feita dentro do fold e limiar sempre derivado do treino. O modelo de referência é o Random Forest, mantido fixo: o que se compara são estratégias de reamostragem, não algoritmos. O resultado origina o **Quadro 1** e as **Tabelas 2 e 3** do TCC.

Com apenas dez clientes Ativos no teste, um único split pode favorecer qualquer estratégia por acaso. Repetir trinta vezes e reportar média com desvio padrão mostra se a vantagem observada é real ou coincidência.

**Split de referência (random_state=42), Tabela 2:**

| Estratégia | N treino | AUC | CV-AUC | MCC | Kappa | F1-Macro | Rec-Ativo | Rec-Churn | Limiar |
|---|---|---|---|---|---|---|---|---|---|
| Sem balanceamento | 942 | 0,8018 | 0,6017 | 0,2983 | 0,2680 | 0,6319 | 0,20 | 0,9912 | 0,75 |
| Undersampling puro (1:1) | 80 | 0,7451 | 0,6080 | 0,1943 | 0,1884 | 0,5930 | 0,30 | 0,9425 | 0,27 |
| SMOTE | 1.804 | 0,7327 | 0,5928 | 0,1253 | 0,1233 | 0,5610 | 0,20 | 0,9469 | 0,44 |
| **Under 3:1 + gaussiano (adotada)** | 240 | **0,8106** | 0,6814 | 0,1931 | 0,1918 | 0,5957 | 0,20 | 0,9735 | 0,38 |

**Robustez em 30 splits estratificados independentes (média ± desvio padrão), Tabela 3:**

| Estratégia | AUC média | MCC médio | Kappa médio | AUC min-max |
|---|---|---|---|---|
| Sem balanceamento | 0,6543 ± 0,0778 | 0,1168 ± 0,1167 | 0,1031 ± 0,1004 | 0,462 a 0,800 |
| Undersampling puro (1:1) | 0,6510 ± 0,0769 | 0,0439 ± 0,0765 | 0,0392 ± 0,0689 | 0,501 a 0,802 |
| SMOTE | 0,6615 ± 0,0660 | 0,1060 ± 0,0761 | 0,0895 ± 0,0696 | 0,539 a 0,810 |
| **Under 3:1 + gaussiano** | **0,6686 ± 0,0831** | 0,1140 ± 0,0804 | 0,1053 ± 0,0726 | 0,500 a 0,822 |

A estratégia adotada lidera em AUC nas duas avaliações, mas o resultado é reportado com honestidade quanto aos limites:

- **As diferenças entre estratégias cabem dentro de um desvio padrão**, o que indica tendência, não superioridade estatisticamente estabelecida.
- **As métricas dependentes de limiar não são comparação direta.** Como o limiar é derivado separadamente em cada estratégia, MCC, Kappa, F1-Macro e os recalls são calculados em pontos de corte distintos. A comparação estritamente pareada é a das colunas AUC e CV-AUC (nota da Tabela 2 do TCC).
- **No split de referência, a ausência de balanceamento vence em MCC, Kappa e F1-Macro**, mas sem ganho na classe minoritária: as duas estratégias acertam a mesma proporção de Ativos (0,20), e a diferença vem só do Recall-Churn (0,9912 contra 0,9735). Na média das trinta divisões a diferença de MCC praticamente desaparece (0,1140 contra 0,1168), e a estratégia adotada tem o maior Kappa médio (0,1053).

A média de 0,6686 nas trinta partições é também a principal referência de desempenho esperado do modelo, retomada na Conclusão do TCC.

---

## Seção 7, Comparação de Algoritmos, Benchmark Completo

### O que faz

Treina **dez algoritmos de Machine Learning**, todos com validação cruzada estratificada de 5 folds e reamostragem executada dentro de cada fold, e avalia cada um no hold-out por um conjunto de métricas. Deriva o ponto de corte individualmente a partir do treino, monta o leaderboard (**Tabela 4** do TCC) e declara o modelo de produção.

### Por que dez algoritmos

Cobrir as principais famílias de classificadores tabulares evita viés de seleção: linear (Regressão Logística), árvore única (Decision Tree), ensembles de bagging (Random Forest, Extra Trees), ensembles de boosting (Gradient Boosting, HistGradient Boosting, AdaBoost), vizinhança (KNN), probabilístico (Naive Bayes) e margem máxima (SVM). Limitar a um único algoritmo permitiria argumentar que outro teria sido melhor; com dez, a comparação é defensável.

### Por que o protocolo de validação foi corrigido

A concepção inicial deste pipeline continha dois vieses, corrigidos após revisão da orientadora:

1. **Reamostragem fora do fold**: a validação cruzada rodava sobre um conjunto já balanceado, espalhando instâncias sintéticas geradas da mesma observação original entre folds distintos e inflando a estimativa (vazamento de informação entre treino e validação). Agora, imputação, padronização e balanceamento são ajustados **dentro** de cada fold, exclusivamente sobre a porção de ajuste; a porção de validação permanece na distribuição original de classes.
2. **Limiar escolhido no teste**: o ponto de corte era escolhido maximizando F1-Macro diretamente contra `y_te`, usando o conjunto de teste duas vezes (uma para ajustar o limiar, outra para reportar a métrica), o que caracteriza viés de seleção na avaliação (CAWLEY; TALBOT, 2010). Agora, o limiar é derivado das **probabilidades out-of-fold do treino** e congelado antes de qualquer contato com o hold-out.

A coluna `CV-AUC vazada` do leaderboard reproduz deliberadamente o procedimento anterior, apenas para dimensionar o efeito da correção, e não deve ser lida como resultado. A queda ordena-se pela capacidade de cada algoritmo de memorizar quase duplicatas:

| Modelo | CV-AUC vazada | CV-AUC correta | Queda |
|---|---|---|---|
| AdaBoost | 0,8623 | 0,5372 | 0,3251 |
| KNN | 0,9135 | 0,5913 | 0,3222 |
| Extra Trees | 0,9434 | 0,6777 | 0,2657 |
| Gradient Boosting | 0,8931 | 0,6695 | 0,2236 |
| Random Forest | 0,9021 | 0,6814 | 0,2207 |
| SVM | 0,8344 | 0,6238 | 0,2106 |
| Decision Tree | 0,8000 | 0,6119 | 0,1881 |
| HistGradient Boosting | 0,8677 | 0,7363 | 0,1314 |
| Logistic Regression | 0,7115 | 0,6326 | 0,0789 |
| Naive Bayes | 0,7024 | 0,6390 | 0,0634 |

O caso do KNN é ilustrativo: a CV-AUC passa de 0,9135 para 0,5913, valor compatível com os 0,5336 do hold-out. A conclusão do TCC (Seção 4.2) é que a diferença entre validação cruzada e hold-out observada na concepção inicial media vazamento entre partições, e não sobreajuste.

### Por que protocolo multi-métrica

Em contextos desbalanceados a acurácia é enganosa (prever sempre Churn dá cerca de 96% de acurácia sem nenhum poder preditivo). De la Cruz Huayanay, Bazán e Russo (2025), em estudo de simulação publicado na *Computational Statistics*, mostraram que AUC e acurácia não distinguem o modelo corretamente especificado do mal especificado em nenhum cenário avaliado, enquanto **MCC, G-Mean e Kappa de Cohen** o fazem de forma consistente. Este trabalho mantém a AUC por comparabilidade e reporta as três métricas recomendadas lado a lado.

A coluna `AP` do leaderboard é a Average Precision da classe **Churn**, cuja linha de base é a própria prevalência (0,958). Sua magnitude absoluta é pouco informativa; o que importa é a comparação entre modelos. A AP da classe Ativo é calculada em célula separada, adiante.

### Sobre a verificação de sobreajuste

A diferença `Δ = |CV-AUC - AUC hold-out|` é **verificada e reportada, mas não usada para descartar modelos**. O maior Δ observado é 0,1347 (Decision Tree), seguido de 0,1293 (Random Forest), ambos abaixo do limite de referência `GAP_MAX = 0,15`. O TCC registra que esse limite é um parâmetro operacional definido antes do benchmark, e não um valor extraído da literatura, e reporta a sensibilidade: sob 0,10, Decision Tree e Random Forest seriam descartados; sob 0,20, o resultado seria idêntico. Como nenhum modelo foi excluído, o critério não participou da escolha do modelo de produção.

### Sobre o limiar

O limiar de cada modelo maximiza o **Kappa de Cohen** (`CRITERIO_LIMIAR = 'kappa'`), critério também adotado por De la Cruz Huayanay, Bazán e Russo (2025). A variante que maximiza F1-Macro é calculada em paralelo (coluna `Thr (F1 alt.)`) como análise de sensibilidade: as duas convergem em oito dos dez modelos, divergindo apenas no SVM (0,37 contra 0,23) e na Árvore de Decisão (0,61 contra 0,01).

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

Sob o treino perfeitamente balanceado (120/120), `class_weight='balanced'` atribui pesos iguais às classes e é, portanto, neutro; o TCC o registra como padrão defensivo.

### Resultado obtido, leaderboard (Tabela 4, ordenado por AUC)

| Modelo | AUC | AP (Churn) | CV-AUC | Δ | F1-Macro | MCC | G-Mean | Kappa | Prec-Ativo | Rec-Ativo | Rec-Churn | Limiar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **0,8106** | **0,9905** | 0,6814 | 0,1293 | 0,5957 | 0,1931 | 0,4412 | 0,1918 | 0,2500 | 0,20 | 0,9735 | 0,38 |
| HistGradient Boosting | 0,7615 | 0,9872 | 0,7363 | 0,0252 | 0,6049 | 0,2143 | **0,5342** | 0,2110 | 0,2143 | 0,30 | 0,9513 | 0,18 |
| Extra Trees | 0,7491 | 0,9860 | 0,6777 | 0,0714 | 0,5649 | 0,1639 | 0,3148 | 0,1370 | 0,3333 | 0,10 | 0,9912 | 0,34 |
| Logistic Regression | 0,7075 | 0,9838 | 0,6326 | 0,0749 | 0,6440 | **0,3517** | 0,4462 | **0,2939** | 0,6667 | 0,20 | 0,9956 | 0,12 |
| SVM | 0,7035 | 0,9833 | 0,6238 | 0,0798 | 0,4920 | 0,0250 | 0,4111 | 0,0197 | 0,0541 | 0,20 | 0,8451 | 0,37 |
| Gradient Boosting | 0,6642 | 0,9796 | 0,6695 | 0,0053 | 0,5339 | 0,0679 | 0,3106 | 0,0678 | 0,1111 | 0,10 | 0,9646 | 0,08 |
| Naive Bayes | 0,6405 | 0,9552 | 0,6390 | 0,0015 | 0,5887 | 0,1778 | 0,4402 | 0,1775 | 0,2222 | 0,20 | 0,9690 | 0,01 |
| AdaBoost | 0,5544 | 0,9694 | 0,5372 | 0,0172 | 0,5725 | 0,2100 | 0,3155 | 0,1547 | 0,5000 | 0,10 | 0,9956 | 0,47 |
| KNN | 0,5336 | 0,9596 | 0,5913 | 0,0577 | 0,5143 | 0,0314 | 0,3063 | 0,0307 | 0,0667 | 0,10 | 0,9381 | 0,09 |
| Decision Tree | 0,4772 | 0,9559 | 0,6119 | 0,1347 | 0,4592 | 0,0178 | 0,4708 | 0,0112 | 0,0484 | 0,30 | 0,7389 | 0,61 |

**Líder por métrica**: AUC e AP, Random Forest; MCC e Kappa, Regressão Logística; G-Mean e Recall-Ativo, HistGradient Boosting. Nenhum modelo acerta mais de 3 dos 10 Ativos do hold-out, e o Recall-Ativo assume só três valores (0,10, 0,20 e 0,30): consequência matemática de dez instâncias na classe minoritária, não falha algorítmica (LUQUE et al., 2019).

**Detecção da classe Ativo no hold-out** (TN = Ativos corretos; FN = Churn classificados como Ativo):

| Modelo | Ativos acertados | Falsos alarmes | Prec-Ativo |
|---|---|---|---|
| HistGradient Boosting | 3/10 | 11 | 0,2143 |
| Decision Tree | 3/10 | 59 | 0,0484 |
| Random Forest | 2/10 | 6 | 0,2500 |
| Logistic Regression | 2/10 | 1 | 0,6667 |

A liderança da Regressão Logística em MCC e Kappa decorre integralmente da redução de falsos negativos na classe majoritária (um caso contra seis do Random Forest), e não de ganho na classe Ativo, onde ela acerta as mesmas duas instâncias.

### Escolha declarada do modelo de produção

A versão anterior do notebook elegia automaticamente o maior ROC-AUC. A versão final fixa a escolha na constante `MODELO_PRODUCAO = 'Random Forest'`, com a justificativa escrita no próprio código e reproduzida na Seção 4.2 do TCC:

- **Sob as três métricas recomendadas**, o modelo indicado **não** seria o Random Forest (4º em MCC e G-Mean, 3º em Kappa). HistGradient Boosting e Regressão Logística empatam em posição média nos três rankings (1,67 cada), o que mostra que essas métricas apontam a inadequação do Random Forest para a decisão binária, mas não desempatam entre si.
- **O critério vem da natureza do entregável.** O produto do trabalho não é uma decisão binária num limiar, e sim uma priorização em quatro faixas de risco (Seção 9), que depende da **ordenação** dos scores. As métricas independentes de limiar (ROC-AUC e AP) medem ordenação, e o Random Forest lidera ambas. A evidência decisiva, porém, é a **captura em cobertura fixa** (Seção 9), que independe de linha de base.
- **Importância nativa.** O `HistGradientBoostingClassifier` não expõe `feature_importances_`, e a análise de interpretabilidade (objetivo específico 4 do TCC) apoia-se na importância por Gini e na estabilidade dessa ordenação entre partições.

O HistGradient Boosting permanece registrado como alternativa indicada caso o objetivo passe a ser maximizar a detecção da classe minoritária em regime de decisão binária, condicionada a calibração prévia de probabilidades (Platt scaling ou regressão isotônica; NICULESCU-MIZIL; CARUANA, 2005). Os três finalistas (Random Forest, HistGradient Boosting e Regressão Logística) são comparados no **Quadro 2** do TCC.

### Curvas ROC, matrizes de confusão e Average Precision por classe

As dez avaliações também são visualizadas em curvas ROC sobrepostas, radar multimétrico, heatmap e um painel de matrizes de confusão; o painel das matrizes origina as **Figuras 7 e 8** do TCC. Para o Random Forest, a matriz no hold-out mostra 2 Ativos corretos, 8 Ativos classificados como Churn, 6 Churn classificados como Ativo e 220 Churn corretos (Recall-Churn 0,97, 220 de 226). No TCC, a classe Churn é a positiva em todas as métricas por classe: falso positivo designa cliente Ativo classificado como Churn.

A célula de Average Precision por classe calcula também a **AP da classe Ativo**, com o score invertido. Contra um baseline de 0,042 (prevalência de Ativos no hold-out):

| Modelo | AP Churn | AP Ativo | Lift sobre o acaso |
|---|---|---|---|
| Naive Bayes | 0,9552 | 0,2701 | 6,37x |
| HistGradient Boosting | 0,9872 | 0,2559 | 6,04x |
| Extra Trees | 0,9860 | 0,2149 | 5,07x |
| Logistic Regression | 0,9838 | 0,1749 | 4,13x |
| Random Forest | 0,9905 | 0,1494 | 3,53x |
| AdaBoost | 0,9694 | 0,1081 | 2,55x |
| SVM | 0,9833 | 0,0893 | 2,11x |
| Gradient Boosting | 0,9796 | 0,0800 | 1,89x |
| KNN | 0,9596 | 0,0608 | 1,43x |
| Decision Tree | 0,9559 | 0,0418 | 0,99x |

O Random Forest tem AP-Ativo inferior ao HistGradient Boosting e à Regressão Logística. O TCC (Seções 2.5 e 4.2) interpreta essa divergência lembrando que a Average Precision pondera de forma desproporcional o extremo inicial da ordenação, região em que a precisão é estimada sobre poucas observações. Por isso a comparação entre finalistas é feita adiante em cobertura fixa, e não pela AP-Ativo isolada.

### Análise do ponto de corte e curva Precision-Recall

Esta célula origina as **Figuras 9 e 10** do TCC (Seção 4.4). O painel esquerdo mostra o Kappa e o F1-Macro calculados sobre as probabilidades out-of-fold do treino, onde o limiar foi de fato escolhido; a curva do teste aparece em cinza apenas como referência e não participa da escolha. O painel direito traz as curvas Precision-Recall por classe, que independem do limiar.

A distribuição das probabilidades do Random Forest no hold-out tem mediana 0,7787 (mín. 0,1121, máx. 0,9714); no treino out-of-fold, mediana 0,7721. A proximidade entre as duas medianas confirma que o limiar derivado do treino transfere-se ao teste sem correção adicional. No limiar padrão de 0,50, 91,9% dos clientes do hold-out seriam classificados como Churn, abaixo da proporção real (95,8%).

O limiar adotado (0,38) produz Kappa de 0,2049 no próprio treino out-of-fold e 0,1918 no hold-out. Se o limiar fosse escolhido diretamente no teste, o Kappa chegaria a 0,2254 no limiar 0,34: a diferença de 0,0336 mede exatamente o viés otimista que o procedimento antigo incorporava.

Classification report do Random Forest no hold-out (limiar 0,38):

```
              precision    recall  f1-score   support
       Ativo       0.25      0.20      0.22        10
       Churn       0.96      0.97      0.97       226
    accuracy                           0.94       236
```

---

## Seção 7.1, Análise das Métricas Robustas (MCC, G-Mean, Kappa)

### O que faz

Ordena os dez modelos por AUC, MCC, G-Mean e Kappa, e mede o grau de concordância entre os rankings pela correlação de Spearman. Corresponde à Seção 4.3 do TCC.

### Por que faz assim

Responde a uma pergunta raramente feita: escolher o modelo por AUC leva ao mesmo resultado que escolher por MCC, G-Mean ou Kappa? A análise dialoga com a recomendação de De la Cruz Huayanay, Bazán e Russo (2025) contra o uso isolado da ROC-AUC, por evidência de natureza distinta: aqueles autores avaliam em simulação a capacidade de cada métrica de identificar o modelo verdadeiro; aqui se observa a divergência entre rankings de modelos em dados reais.

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

Os quatro critérios elegem **três vencedores distintos**: Random Forest pela AUC, Regressão Logística por MCC e por Kappa, HistGradient Boosting por G-Mean (e também por Recall-Ativo). Nenhum modelo ocupa a primeira posição em mais de dois dos quatro critérios, e o líder por AUC figura apenas em 4º no ranking de G-Mean.

**Conclusão do notebook**: os rankings são **divergentes** em pelo menos uma métrica, e a seleção por ROC-AUC **não é confirmada** pelas métricas robustas. Isso não invalida a escolha do Random Forest, que se apoia na natureza do entregável (Seção 7), mas confirma empiricamente que a escolha não pode ser lida de uma única métrica. Ressalva: os coeficientes são estimados sobre apenas dez modelos e indicam tendência, não estimativa populacional.

---

## Seção 8, Interpretabilidade, Feature Importance (Gini)

### O que faz

Extrai do Random Forest treinado a importância de cada uma das onze features via impureza de Gini: a soma ponderada da redução de impureza que a feature produz em todos os splits de todas as árvores (BREIMAN, 2001). Origina a **Tabela 6** e as **Figuras 11 e 12** do TCC (Seção 4.6).

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

**Top 3 (frequência, recência e diversidade de categorias) respondem por 47,26% da importância total**, e o Top 5 por 62,52% (calculados sobre os valores sem arredondamento).

**Achados-chave**:

- **`frequency` como principal driver** (23,19%): confirma a intuição RFM clássica, clientes que compram mais vezes têm menor propensão ao churn
- **`n_categories` em 3º lugar** (11,88%), em **empate técnico** com `recency_days` (12,19%, diferença de 0,3 ponto percentual): a ordem exata entre 2º e 3º pode inverter entre versões de biblioteca, mas a composição do trio dominante é estável. A diversidade de categorias como indicador de engajamento é pouco documentada nos estudos anteriores sobre o Olist, incluindo Matuszelański e Kopczewska (2022)
- **`avg_review_score` apenas em 6º lugar** (6,99%): confirma a observação da EDA de que o problema não é qualidade do serviço, e sim engajamento
- **`monetary` em 7º** apesar de discriminar as classes na EDA: o TCC sugere que seu poder preditivo é em parte absorvido pela frequência, com a qual mantém dependência direta

---

## Seção 8.5, Análise Comparativa de Representatividade, PSI e KS-Test

### O que faz

Compara a distribuição das 1.128 observações Churn originais com o subconjunto efetivamente retido no treino após o undersampling (902 no treino antes do balanceamento, reduzidas a 120 pela razão 3:1), usando duas métricas complementares: **PSI** (Population Stability Index) e **KS-test** (Kolmogorov-Smirnov, duas amostras). Origina a **Tabela 5** do TCC (Seção 4.5).

### Por que faz assim

Após o undersampling aleatório, surge uma questão metodológica que a maioria dos estudos aplicados sobre o Olist **ignora**: o subconjunto de Churn que foi para o treino representa fielmente a classe Churn completa, ou o sorteio produziu uma amostra distorcida? Se o undersampling tivesse selecionado uma subamostra enviesada (por exemplo, predominantemente clientes de um estado, ou de valor muito baixo), o modelo teria aprendido padrões que não generalizam.

**Critérios de interpretação**:

| Métrica | Faixa | Significado |
|---------|-------|-------------|
| PSI < 0,10 | Distribuições similares | Subconjunto representa bem a população |
| PSI 0,10 a 0,20 | Atenção | Algum deslocamento; investigar |
| PSI > 0,20 | Distorção significativa | Subconjunto não representa a população |
| KS p > 0,05 | Não rejeitar H0 | Sem evidência de diferença entre as distribuições |
| KS p <= 0,05 | Rejeitar H0 | Distribuições significativamente diferentes |

### Como faz

```python
from scipy.stats import ks_2samp
idx_churn_retido = tr_idx[maj_keep_idx]    # posições originais em rfm
for feature in features_numericas:
    psi = calcular_psi(churn_completo[feature], churn_retido[feature])
    ks_stat, p_valor = ks_2samp(churn_completo[feature], churn_retido[feature])
```

A função de PSI trata variáveis discretas de baixa cardinalidade (como `frequency`) usando contagem por valor único em vez de quantis, evitando faixas vazias que distorceriam o índice. A conclusão é emitida por lógica condicional, não escrita de antemão.

### Resultado obtido

Churn completo: 1.128 clientes. Churn no treino, antes do balanceamento: 902. Churn retido após o undersampling: 120 (10,6% da classe completa, 13,3% da parcela presente no treino).

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

**8 de 9 features representativas, 1 em atenção (`avg_installments`), nenhuma distorcida.** Mesmo `avg_installments` tem p-valor de 0,1818 no KS-test, sem rejeição estatística.

**Conclusão e seus limites, conforme o TCC**:

- O peso da conclusão recai sobre o **PSI**. O KS entra como verificação complementar, e **não rejeitar a igualdade não equivale a demonstrá-la**: com 120 observações contra 1.128, a potência do teste é limitada.
- A verificação alcança o **undersampling da classe majoritária**, não a expansão sintética da classe Ativo. Com quarenta observações originais, os testes teriam potência insuficiente para distinguir a distribuição sintética da original; essa validação permanece entre os trabalhos futuros.
- A Seção 4.5.1 do TCC discute a alternativa de seleção orientada por informatividade (*one-sided selection*; KUBAT; MATWIN, 1997), registrada como trabalho futuro para bases maiores.

---

## Seção 8.6, Análise SHAP, Interpretabilidade Avançada

### O que faz

Calcula os **valores SHAP** (SHapley Additive exPlanations) via `TreeExplainer` para o Random Forest, e produz quatro análises: beeswarm com ranking, SHAP médio por classe real, dependence plots das principais features e waterfall plots de clientes típicos. Origina as **Figuras 13 a 19** do TCC (Seções 4.6 e 4.6.1).

### Por que faz assim

A importância via Gini (Seção 8) responde quais features importam. SHAP responde em que direção cada feature influencia as predições e em quais limiares, dimensão essencial para tradução em recomendações de negócio. Os valores SHAP, fundamentados na teoria dos jogos cooperativos de Shapley (1953), são o método de referência atual para explicabilidade local e global (LUNDBERG; LEE, 2017), e o TreeSHAP (LUNDBERG et al., 2020) calcula valores exatos para modelos de árvore, sem amostragem aproximada.

### Como faz

```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_te_sc)   # 236 clientes do hold-out x 11 features
```

Um bloco condicional padroniza o retorno de `shap_values`, que varia entre versões da biblioteca (lista de arrays ou array 3D).

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

As quatro features mais relevantes ocupam posições **idênticas** nos dois rankings, e apenas duas das onze divergem em mais de uma posição. A exceção informativa é `top_category`, que sobe da 8ª posição no Gini para a 5ª no SHAP, coerente com o viés conhecido do Gini contra variáveis categóricas.

**Direção dos efeitos** (correlação entre valor da feature e contribuição SHAP):

| Feature | Correlação | Leitura |
|---|---|---|
| `frequency` | -0,862 | Mais pedidos empurra a predição para Ativo (efeito protetor) |
| `recency_days` | +0,895 | Mais dias sem comprar empurra a predição para Churn |
| `n_categories` | -0,648 | Mais categorias empurra a predição para Ativo (efeito protetor) |

**Dependence plots** (Figuras 15 a 17, arquivo `apendiceA_dependence.png`), conforme a leitura do TCC:

- `frequency`: dois agrupamentos disjuntos. Clientes com dois pedidos recebem contribuição positiva para o churn (entre +0,08 e +0,15); com três ou mais, contribuição negativa. É um limiar de decisão, não uma tendência gradual.
- `recency_days`: relação monotônica crescente, aproximadamente sigmoide, com a transição de negativo para positivo em torno da média da distribuição padronizada.
- `n_categories`: repete a descontinuidade de `frequency`, com separação nítida entre uma ou duas categorias e três ou mais.

**Waterfall plots** (Figuras 18 e 19, arquivos `apendiceA_waterfall_ativo.png` e `apendiceA_waterfall_churn.png`), para o cliente mais próximo do centroide de cada classe no hold-out:

- **Cliente típico Churn** (índice 159): P(Churn) = 0,841, consistente com a classe real. O deslocamento vem principalmente de frequência (+0,12), diversidade de categorias (+0,08) e parcelamento médio (+0,07).
- **Cliente típico Ativo** (índice 108): P(Churn) = 0,613, acima do limiar de 0,38, portanto **classificado incorretamente como Churn**. Tem frequência e diversidade de categorias abaixo da média, e as contribuições que o afastam do churn (parcelamento médio -0,06, recência -0,03) não compensam. O TCC usa o caso para ilustrar o Recall-Ativo de 0,20: o perfil médio de quem retorna não é suficientemente distinto do perfil de quem abandona, limitação do tamanho da classe minoritária e não da especificação do modelo.

**Perfil do cliente que retorna** (Seção 4.6.1 do TCC): invertendo a leitura das contribuições SHAP, o cliente com maior propensão a retornar compra com mais frequência, mantém recência baixa e compra em mais categorias distintas. Parcelamento médio elevado associa-se a maior risco, possivelmente por sinalizar compras pontuais de maior ticket. O perfil deve ser lido como hipótese robusta a confirmar em bases maiores, dada a classe Ativo de cerca de 50 clientes.

> **Ressalva metodológica importante.** A concordância entre Gini e SHAP **não constitui validação independente**: a importância de Gini mede redução de impureza nas árvores já ajustadas, e o TreeSHAP percorre exatamente essas mesmas árvores. A concordância é esperada por construção. É por isso que existem as duas análises seguintes: a comparação SHAP entre modelos de naturezas distintas e a estabilidade da ordenação entre partições.

---

## Importância |SHAP| Comparada entre os Três Modelos (Tabela 7 do TCC)

### O que faz

Recalcula a importância média absoluta dos valores SHAP para o Random Forest, o HistGradient Boosting e a Regressão Logística sobre o mesmo hold-out (236 clientes, 11 features), e monta a tabela de posições e participações relativas que origina a **Tabela 7** do TCC. Salva `tabela7_shap_ranking.csv`.

### Por que faz assim

A importância por Gini não existe no HistGradient Boosting e não se aplica à Regressão Logística. O SHAP é o único estimador calculável para os três, portanto o único terreno comum para verificar se a hierarquia de importância é propriedade dos dados ou artefato do estimador escolhido.

**Ressalva de escala.** O TreeExplainer aplicado ao Random Forest devolve contribuições em espaço de probabilidade; HistGradient Boosting e Regressão Logística devolvem contribuições em log-odds. As magnitudes absolutas **não** são comparáveis entre modelos. Comparáveis são a **posição** no ranking e a **participação relativa** (|φ| médio da feature dividido pela soma dos |φ| médios das onze features do mesmo modelo).

### Como faz

| Modelo | Origem dos valores SHAP |
|---|---|
| Random Forest | TreeExplainer (probabilidade), reaproveitado da Seção 8.6 |
| HistGradient Boosting | TreeExplainer (log-odds); cai para PermutationExplainer se falhar |
| Logistic Regression | LinearExplainer (log-odds); cai para a forma fechada `β_j · (x_j − E[x_j])` se falhar |

Na execução de referência, os três caminhos primários funcionaram.

### Resultado obtido

| Feature | Pos. RF | Pos. HistGB | Pos. RL | % RF | % HistGB | % RL |
|---|---|---|---|---|---|---|
| frequency | 1ª | 1ª | 3ª | 27,0 | 19,4 | 13,9 |
| recency_days | 2ª | 2ª | 2ª | 15,7 | 13,4 | 19,0 |
| n_categories | 3ª | 6ª | 4ª | 15,0 | 8,2 | 10,6 |
| avg_installments | 4ª | 4ª | 1ª | 13,2 | 11,7 | 21,0 |
| top_category | 5ª | 5ª | 7ª | 5,9 | 9,5 | 6,1 |
| avg_review_score | 6ª | 9ª | 8ª | 5,0 | 4,9 | 5,8 |
| avg_delivery_days | 7ª | 3ª | 6ª | 4,7 | 11,8 | 6,4 |
| monetary | 8ª | 7ª | 9ª | 4,4 | 6,4 | 4,7 |
| avg_freight | 9ª | 11ª | 11ª | 3,5 | 4,3 | 2,5 |
| customer_state | 10ª | 8ª | 10ª | 2,9 | 5,9 | 3,7 |
| avg_ticket | 11ª | 10ª | 5ª | 2,7 | 4,6 | 6,4 |

**Convergência entre os rankings** (Spearman sobre as importâncias):

| Par | rho | p | Divergem > 1 posição |
|---|---|---|---|
| RF x HistGB | 0,8000 | 0,0031 | 5 de 11 |
| RF x RL | 0,7091 | 0,0146 | 6 de 11 |
| HistGB x RL | 0,7091 | 0,0146 | 8 de 11 |

**Leitura (Seção 4.6 do TCC)**:

- **A recência ocupa a 2ª posição nos três modelos**, sem exceção: é o achado mais robusto da análise de interpretabilidade.
- **A frequência lidera nos dois ensembles** e cai para 3ª no modelo linear, coerente com o efeito descontínuo (dois pedidos contra três ou mais) visto nos dependence plots, que um modelo linear captura só parcialmente.
- **`avg_installments` é 1ª na Regressão Logística**, sugerindo efeito predominantemente monotônico.
- **`avg_delivery_days` é 3ª no HistGradient Boosting** e 7ª no Random Forest, sugerindo sensibilidade do boosting a limiares do tempo de entrega.
- **`n_categories` cai para 6ª no HistGradient Boosting.** A composição do trio dominante é confirmada por dois dos três estimadores, e não pelos três.

---

## Estabilidade da Importância entre Partições e Comparação entre Estimadores

*(corresponde à Seção 4.6 do TCC; no PDF final, a Seção 4.6.1 é o perfil do cliente que retorna)*

### O que faz

Substitui a alegação de "validação cruzada entre Gini e SHAP" por uma evidência efetivamente independente do estimador: a estabilidade da ordenação por Gini em **30 partições estratificadas independentes** do conjunto de treino. Em paralelo, calcula a **importância por permutação** sobre o hold-out, para o Random Forest e, como controle, para o HistGradient Boosting.

### Por que faz assim

Gini e SHAP compartilham a mesma fonte de informação (a estrutura das árvores ajustadas): concordarem entre si não prova nada de novo. Repetir o treino em partições diferentes dos dados testa se a hierarquia é artefato de uma única divisão.

### Como faz

Para cada semente de 0 a 29: split estratificado 80/20, imputação e escala ajustadas no treino, `balance_dataset` com `RATIO_MAJ = 3`, Random Forest com os hiperparâmetros de produção, e registro da posição de cada feature no ranking de Gini. A importância por permutação usa `n_repeats=30` e `scoring='roc_auc'` sobre o hold-out.

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
| avg_ticket | 7,27 | 4 | 11 | 0 |
| avg_installments | 7,30 | 3 | 11 | 1 |
| avg_freight | 8,53 | 5 | 11 | 0 |
| top_category | 8,97 | 5 | 11 | 0 |
| customer_state | 9,70 | 5 | 11 | 0 |

`frequency` ocupa o 1º lugar em todas as 30 partições. `n_categories` fica no Top 3 em 29 de 30. `recency_days` fica no Top 3 em 22 de 30, oscilando entre a 2ª e a 5ª posição. A variável seguinte, `monetary`, entra no Top 3 em apenas 4 partições, o que delimita com clareza o trio dominante. **É esta a evidência de estabilidade citada no TCC.** Ela também confirma o empate técnico da Seção 8: embora a recência esteja 0,3 ponto à frente na partição de referência, a diversidade de categorias tem posição média melhor nas trinta partições (2,30 contra 3,17).

**Importância por permutação (Random Forest, hold-out):**

| Feature | Gini (rank) | Permutação média ± dp | Rank permutação |
|---|---|---|---|
| frequency | 1 | 0,0339 ± 0,0148 | 4 |
| recency_days | 2 | 0,0721 ± 0,0508 | 2 |
| n_categories | 3 | 0,0235 ± 0,0219 | 7 |
| avg_installments | 4 | 0,0788 ± 0,0449 | 1 |
| avg_delivery_days | 5 | 0,0239 ± 0,0192 | 6 |
| avg_review_score | 6 | 0,0348 ± 0,0237 | 3 |
| monetary | 7 | 0,0250 ± 0,0140 | 5 |
| top_category | 8 | -0,0091 ± 0,0104 | 11 |
| avg_freight | 9 | 0,0183 ± 0,0134 | 8 |
| avg_ticket | 10 | 0,0136 ± 0,0117 | 9 |
| customer_state | 11 | -0,0059 ± 0,0130 | 10 |

**Leitura correta da permutação** (revisada em relação à versão de agosto deste guia): **o estimador não é informativo neste regime amostral.** Com apenas dez instâncias Ativas no hold-out, o desvio padrão de cada estimativa fica entre 44% e 93% da respectiva média nas nove variáveis de média positiva, e as duas restantes têm média negativa, valor sem interpretação substantiva. A imprecisão dissolve a própria ordenação que o estimador produz: `avg_installments` (0,0788 ± 0,0449) e `recency_days` (0,0721 ± 0,0508), 1ª e 2ª colocadas, têm intervalos amplamente sobrepostos. A divergência em relação ao Gini **não refuta** a ordenação por Gini, mas impede afirmar robustez da hierarquia ao método de estimação. Foi por isso que a estabilidade entre partições, e não a permutação, foi adotada como evidência.

**Controle no HistGradient Boosting**: a permutação coloca `avg_delivery_days` em 1º (0,0689) e `n_categories` em 10º (0,0026), com `customer_state` negativa. O notebook registra ainda que o `HistGradientBoostingClassifier` não expõe `feature_importances_`, o que inviabiliza a análise de estabilidade por Gini para esse modelo; replicá-la exigiria usar a permutação, que acabou de se mostrar não informativa.

---

## Seção 9, Score de Churn e Segmentação de Risco

### O que faz

Aplica o Random Forest treinado a toda a base de 1.178 clientes, atribuindo a cada um uma **probabilidade contínua** P(Churn), e segmenta os clientes em quatro faixas de risco com cortes fixos (0,25 / 0,50 / 0,75). Em seguida valida a segmentação pela taxa de churn observada, compara a distribuição dos scores entre os três finalistas, repete tudo fora da amostra e mede a captura em cobertura fixa. Corresponde às Seções 4.2 e 4.7 do TCC, com a **Tabela 8** e a **Figura 20**.

### Por que faz assim

**Reposicionamento metodológico**: a classificação binária tem qualidade limitada neste dataset (MCC de 0,19 para o Random Forest), consequência direta do tamanho reduzido da classe Ativo. Por isso, o entregável principal não é uma decisão sim/não, e sim uma probabilidade contínua que ordena os clientes por risco. Em vez de forçar uma decisão binária com support de 10 amostras, a segmentação em faixas permite priorizar investimento até o limite do orçamento de retenção, o que é economicamente preferível à reaquisição (REICHHELD; SCHEFTER, 2000).

**Por que esses cortes?** Os limiares de 25%, 50% e 75% são **cortes nominais** da escala de probabilidade, adotados por interpretabilidade e não otimizados sobre os dados; sua adequação é aferida a posteriori pela taxa de churn observada em cada faixa (TCC, Seção 4.7). Não confundir com o limiar de classificação (0,38 no Random Forest), que é parâmetro da regra de decisão binária. A segmentação por risco para campanhas de retenção segue Kotler e Keller (2016).

### Como faz

```python
rfm['churn_proba']   = rf_model.predict_proba(X_all)[:, 1]
BINS_RISCO   = [0, .25, .50, .75, 1.001]
LABELS_RISCO = ['🟢 Baixo', '🟡 Médio', '🟠 Alto', '🔴 Crítico']
rfm['risk_segment']  = pd.cut(rfm['churn_proba'], bins=BINS_RISCO, labels=LABELS_RISCO)
```

`pd.cut` é fechado à direita: os intervalos reais são `(0; 0,25]`, `(0,25; 0,50]`, `(0,50; 0,75]` e `(0,75; 1]`. Portanto Crítico é `P > 75%` e Baixo é `P <= 25%`. Os cortes são constantes para que a Figura 20 e a Tabela 8 leiam da mesma fonte e não possam divergir.

### Resultado obtido, perfil por segmento (Tabela 8)

| Segmento | Clientes | % da base | Recência média (dias) | Monetário médio (R$) | Review médio | P(Churn) média |
|----------|----------|---|---|---|---|---|
| 🟢 Baixo | 27 | 2,3% | 72,52 | 418,49 | 4,17 | 0,15 |
| 🟡 Médio | 101 | 8,6% | 80,72 | 252,77 | 4,35 | 0,40 |
| 🟠 Alto | 365 | 31,0% | 95,78 | 307,11 | 4,08 | 0,65 |
| 🔴 Crítico | 685 | 58,1% | 147,95 | 302,33 | 4,26 | 0,85 |

A Tabela 8 do TCC traz critério, clientes, recência média e a ação recomendada: Crítico, ação urgente com cupons e contato direto; Alto, ofertas e descontos personalizados; Médio, campanhas de reengajamento leve; Baixo, cross-sell e manutenção de engajamento.

### Figura 20, distribuição dos scores nos três modelos

A célula gera `figura9_scores.png` (nome de arquivo histórico; é a **Figura 20** do PDF final), com um painel por modelo e as três linhas de corte. Uma conferência interna verifica que o painel do Random Forest reproduz `rfm['churn_proba']` (divergência máxima de 2,22e-16).

### Validação da segmentação pela taxa de churn observada

A tabela acima mostra apenas quantos clientes caem em cada faixa; isso não demonstra que as faixas separam risco de fato. A prova está no rótulo `churn` real, externo ao score. Recência, monetário e demais colunas da Tabela 8 **não** servem a esse fim, porque são features do modelo.

| Faixa | n | Churn | Ativo | Taxa de churn | Lift sobre Ativo |
|---|---|---|---|---|---|
| Baixo | 27 | 6 | 21 | 22,2% | 18,32x |
| Médio | 101 | 80 | 21 | 79,2% | 4,90x |
| Alto | 365 | 358 | 7 | 98,1% | 0,45x |
| Crítico | 685 | 684 | 1 | 99,9% | 0,03x |

Monotonicidade estrita confirmada (Baixo < Médio < Alto < Crítico), com amplitude de **77,6 pontos percentuais**. Como a taxa base de churn já é 95,8%, o lift sobre Churn tem teto natural em torno de 1,04x; a leitura relevante está no lift sobre **Ativo**.

**Atenção à composição da base completa**: ela contém as 160 linhas efetivamente usadas no ajuste, entre elas os 40 clientes Ativos do treino, que caem **todos** nas faixas Baixo e Médio. Por isso o recorte Baixo + Médio (42 de 50 Ativos em 10,9% da base, lift 7,73x) mede em parte memorização e **não** é o número defendido. O recorte que sobrevive fora da amostra é a **exclusão da faixa Crítico** (ver a seguir), reportado nas Seções 4.2, 4.7 e na Conclusão do TCC.

### Diagnósticos comparativos: Random Forest, HistGradient Boosting e Regressão Logística

O HistGradient Boosting vence o Random Forest em MCC, Kappa, G-Mean, F1-Macro e Recall-Ativo, e a Regressão Logística lidera MCC e Kappa. Os diagnósticos a seguir comparam a **distribuição dos scores** dos três. Eles avaliam distribuição, não desempenho: a base completa contém observações de treino, e qualquer AUC calculado nela seria in-sample.

**Dispersão dos scores (base completa, n = 1.178):**

| Modelo | Mín | P25 | Mediana | P75 | Máx | IQR | Limiar |
|---|---|---|---|---|---|---|---|
| Random Forest | 0,0033 | 0,6545 | 0,7781 | 0,8601 | 0,9934 | 0,2056 | 0,38 |
| HistGradient Boosting | 0,0006 | 0,7927 | 0,9621 | 0,9915 | 0,9999 | 0,1988 | 0,18 |
| Logistic Regression | 0,0009 | 0,4204 | 0,6000 | 0,7818 | 0,9988 | 0,3614 | 0,12 |

**Ocupação das quatro faixas, com os mesmos cortes da Tabela 8:**

| Modelo | Escopo | Baixo | Médio | Alto | Crítico |
|---|---|---|---|---|---|
| Random Forest | base completa | 2,3% | 8,6% | 31,0% | 58,1% |
| HistGradient Boosting | base completa | 9,6% | 5,6% | 7,2% | 77,6% |
| Logistic Regression | base completa | 5,8% | 31,2% | 33,4% | 29,6% |
| Random Forest | hold-out | 1,7% | 6,4% | 30,9% | 61,0% |
| HistGradient Boosting | hold-out | 8,1% | 5,9% | 6,4% | 79,7% |
| Logistic Regression | hold-out | 5,1% | 30,1% | 31,8% | 33,1% |

O HistGradient Boosting **satura os scores junto do extremo superior** da escala (mediana 0,9621), esvaziando as faixas intermediárias e concentrando 77,6% da base (79,7% do hold-out) em Crítico: sua escala comporta essencialmente uma faixa útil. O limiar baixo do HistGB (0,18) não indica probabilidades comprimidas perto de zero; é o oposto: como praticamente todo score excede esse valor, a varredura converge para um corte baixo. Esse deslocamento das probabilidades para a classe majoritária do treino é o efeito descrito por Dal Pozzolo et al. (2015), que preserva a **ordenação** relativa dos clientes (por isso o AUC do HistGB não colapsa). O padrão se reproduz no hold-out, o que afasta a hipótese de artefato das linhas de treino.

**A dispersão da escala, porém, não é critério de escolha isolado.** A Regressão Logística tem a distribuição mais uniforme das três e, como mostra a curva de captura adiante, é a menos eficaz em identificar clientes recuperáveis por unidade de cobertura. A vantagem do Random Forest está na **ordenação**, não na dispersão.

### Diagnóstico restrito ao hold-out

Repete a validação apenas sobre os 236 clientes nunca vistos pelo modelo, separando os 50 Ativos por origem.

| Origem | Clientes | Ativos |
|---|---|---|
| Hold-out (nunca visto) | 236 | 10 |
| Treino usado no ajuste | 160 | 40 |
| Treino descartado no undersampling | 782 | 0 |
| **Total** | 1.178 | 50 |

Todo Ativo do treino é preservado; apenas Churn é reduzido pelo undersampling. Os 40 Ativos vistos no ajuste caem 20 no Baixo e 20 no Médio; os 10 do hold-out caem 1 no Baixo, 1 no Médio, 7 no Alto e 1 no Crítico.

**Faixas no hold-out, Random Forest (n = 236, Ativos = 10):**

| Faixa | n | Churn | Ativo | Taxa de churn | Lift Ativo |
|---|---|---|---|---|---|
| Baixo | 4 | 3 | 1 | 75,0% | 5,90x |
| Médio | 15 | 14 | 1 | 93,3% | 1,57x |
| Alto | 73 | 66 | 7 | 90,4% | 2,26x |
| Crítico | 144 | 143 | 1 | 99,3% | 0,16x |

A progressão se mantém no hold-out (75,0% no Baixo, 99,3% no Crítico), com uma inversão entre Médio e Alto atribuível ao número reduzido de observações por faixa (15 e 73 clientes), não a uma quebra real de monotonicidade.

**Recorte Baixo + Médio, base completa contra hold-out** (não sobrevive fora da amostra):

| Modelo | Base completa | Hold-out |
|---|---|---|
| Random Forest | 42/50, 10,9%, lift 7,73x | 2/10, 8,1%, lift 2,48x |
| HistGradient Boosting | 43/50, 15,2%, lift 5,66x | 3/10, 14,0%, lift 2,15x |
| Logistic Regression | 32/50, 37,0%, lift 1,73x | 5/10, 35,2%, lift 1,42x |

No hold-out, a diferença entre Random Forest e HistGradient Boosting corresponde a um único cliente e não autoriza afirmar superioridade: o TCC não usa esse recorte para desempatar.

**Recorte "fora da faixa Crítico"**, o número efetivamente citado nas Seções 4.2, 4.7 e na Conclusão:

| Modelo | Hold-out (Ativos / cobertura / lift) | Base completa (Ativos / cobertura / lift) |
|---|---|---|
| Random Forest | 9/10, 39,0% (92 de 236), 2,31x | 49/50, 41,9%, 2,34x |
| HistGradient Boosting | 3/10, 20,3%, 1,48x | 43/50, 22,4%, 3,84x |
| Logistic Regression | 10/10, 66,9%, 1,49x | 46/50, 70,4%, 1,31x |

O lift do Random Forest é estável entre base completa e hold-out (2,34x contra 2,31x), o que indica não haver, nesse recorte, ganho atribuível à memorização das linhas de treino. No HistGradient Boosting, a base completa (3,84x) supera muito o hold-out (1,48x), sugerindo que parte da vantagem aparente na base completa vem das linhas vistas no ajuste. A Regressão Logística alcança todos os Ativos, mas ao custo de abordar dois terços do hold-out, cobertura que descaracteriza o propósito da priorização.

### Curva de captura em cobertura fixa

**Por que existe.** O recorte "fora do Crítico" entrega frações diferentes da base em cada modelo (39,0%, 20,3% e 66,9%), então o lift mistura a qualidade da ordenação com o ponto onde o corte de 0,75 caiu em cada distribuição. Fixando a cobertura, os três modelos passam a ser avaliados sobre o mesmo orçamento de campanha, e o que sobra na comparação é só a **ordenação**. É a evidência decisiva da escolha do modelo de produção no TCC e uma das contribuições listadas na Conclusão.

**Como faz.** Ordena os clientes do menor para o maior score de churn e conta quantos Ativos reais entram nos X% de menor score, para X de 10% a 60%. Calcula também a cobertura mínima para alcançar 90% e 100% dos Ativos. Salva `captura_cobertura_fixa.csv`.

**Ativos capturados no hold-out (de 10):**

| Cobertura | Random Forest | HistGradient Boosting | Logistic Regression |
|---|---|---|---|
| 10% | 3 (lift 2,95x) | 3 (2,95x) | 2 (1,97x) |
| 20% | 4 (1,97x) | 3 (1,48x) | 2 (0,98x) |
| 30% | 8 (2,66x) | 6 (1,99x) | 5 (1,66x) |
| 40% | 9 (2,24x) | 8 (1,99x) | 7 (1,74x) |
| 50% | 10 (2,00x) | 9 (1,80x) | 8 (1,60x) |
| 60% | 10 (1,66x) | 10 (1,66x) | 10 (1,66x) |

**Sínteses independentes da cobertura escolhida (hold-out):**

| Modelo | AP-Ativo | ROC-AUC | Cobertura p/ 90% dos Ativos | Cobertura p/ 100% dos Ativos |
|---|---|---|---|---|
| Random Forest | 0,1494 | 0,8106 | **35,6%** | **44,9%** |
| HistGradient Boosting | 0,2559 | 0,7615 | 49,6% | 51,3% |
| Logistic Regression | 0,1749 | 0,7075 | 52,5% | 58,5% |

O Random Forest **lidera ou empata em todas as coberturas** de 10% a 60%, e alcança 90% dos Ativos abordando 35,6% do hold-out. Abaixo de 30% nenhum modelo passa de quatro acertos, e a partir de 60% os três alcançam todos os Ativos: a escolha do modelo só tem efeito operacional no intervalo intermediário, justamente onde o Random Forest tem vantagem. Na Regressão Logística, o lift de 0,98x em 20% de cobertura é indistinguível do acaso. O valor de 35,6% é o parâmetro de dimensionamento de campanha proposto na Seção 4.7 do TCC.

> **Ressalva a manter.** O hold-out tem apenas 10 Ativos: cada unidade capturada vale 10 pontos percentuais de recall, e diferenças de um cliente entre modelos não são conclusivas. A mesma curva é impressa na base completa como referência de tendência, mas contém observações de treino e não mede desempenho preditivo.

### Consolidação numérica dos três modelos

A última célula da seção reúne num único bloco todo número que o texto cita sobre os três finalistas: desempenho no hold-out (Bloco 1), líder por métrica entre os dez (Bloco 2), dispersão dos scores (Bloco 3), ocupação das faixas na base completa e no hold-out (Bloco 4) e o recorte "fora da faixa Crítico" (Bloco 5). Salva `consolidado_v8_tres_modelos.csv`. Serve como fonte única de conferência: nenhum valor do TCC sobre os três modelos deve divergir dela.

| Métrica (hold-out) | Random Forest | HistGradient Boosting | Logistic Regression |
|---|---|---|---|
| ROC-AUC | 0,8106 | 0,7615 | 0,7075 |
| AP (Churn) | 0,9905 | 0,9872 | 0,9838 |
| CV-AUC (dp) | 0,6814 (0,0661) | 0,7363 (0,0487) | 0,6326 (0,0466) |
| \|CV - hold-out\| | 0,1293 | 0,0252 | 0,0749 |
| MCC | 0,1931 | 0,2143 | 0,3517 |
| Kappa | 0,1918 | 0,2110 | 0,2939 |
| G-Mean | 0,4412 | 0,5342 | 0,4462 |
| Rec-Ativo / Rec-Churn | 0,20 / 0,9735 | 0,30 / 0,9513 | 0,20 / 0,9956 |
| TN/FP/FN/TP | 2/8/6/220 | 3/7/11/215 | 2/8/1/225 |
| Limiar | 0,38 | 0,18 | 0,12 |
| Lift fora do Crítico (hold-out / base) | 2,31 / 2,34 | 1,48 / 3,84 | 1,49 / 1,31 |

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

**Todos os arquivos gerados pelo notebook** (na mesma pasta, apenas quando as células correspondentes são executadas):

| Arquivo | Origem | Uso no TCC |
|---|---|---|
| `olist_churn_predictions.csv` | Seção 10 | Entregável operacional |
| `apendiceA_dependence.png` | Seção 8.6 | Figuras 15 a 17 |
| `apendiceA_waterfall_ativo.png`, `apendiceA_waterfall_churn.png` | Seção 8.6 | Figuras 18 e 19 |
| `tabela7_shap_ranking.csv` | SHAP nos três modelos | Tabela 7 |
| `figura9_scores.png` | Seção 9 | Figura 20 |
| `captura_cobertura_fixa.csv` | Seção 9 | Seções 4.2 e 4.7 |
| `consolidado_v8_tres_modelos.csv` | Seção 9 | Conferência dos números dos três modelos |
| `figura1_pipeline.png`, `figura2_janelas.png` | Figuras de fechamento | Figuras 1 e 2 |

---

## Figuras de Fechamento (Figura 1 e Figura 2 do TCC)

Duas células finais desenham os diagramas de síntese do TCC, com **todos os rótulos derivados do estado real do notebook** (número de tabelas, de features, de algoritmos, de folds, tamanhos de treino e hold-out, datas efetivas das janelas), em vez de valores escritos à mão. Essa decisão corrige um problema real de uma versão anterior, em que a figura afirmava "5 tabelas" e "jan a jun de 2018" enquanto o código já usava seis tabelas e uma janela de predição de cinco meses; com valores derivados, essa divergência se torna estruturalmente impossível.

**Figura 1, pipeline metodológico**: tabelas relacionais = 6; observação = set/2016 a dez/2017 (16 meses); predição = jan a mai/2018 (5 meses); features = 11; filtro = `frequency >= 2`; algoritmos = 10; treino 942 / hold-out 236; folds da validação cruzada = 5; limite de referência do gap CV-hold-out = 0,15. A etapa (3) define a estratégia de balanceamento e a etapa (4) a aplica dentro de cada fold, conforme a nota da Figura 1 no TCC.

**Figura 2, janela temporal dupla**: o início da observação vem do primeiro pedido real da base (`obs_orders['order_purchase_timestamp'].min()`), não da constante `OBS_START`; as larguras das barras são proporcionais à duração efetiva de cada janela. Pedidos na janela de observação: 43.695. Pedidos na janela de predição: 34.174.

Cada célula imprime ao final uma tabela de conferência que mapeia cada valor à seção correspondente do TCC.

---

## Síntese Final

### Os principais achados em uma página

**Sobre o modelo**:
- Random Forest adotado como modelo de produção por decisão declarada (`MODELO_PRODUCAO`), com ROC-AUC = 0,8106 e AP-Churn = 0,9905 no hold-out, líder nas duas métricas independentes de limiar
- O desempenho esperado em uso é o indicado pela CV-AUC (0,6814) e pela média de trinta partições (0,6686 ± 0,0831): a partição de referência situa-se na faixa superior da distribuição amostral
- Não é o melhor modelo sob as métricas de decisão binária: a Regressão Logística lidera MCC (0,3517) e Kappa (0,2939), o HistGradient Boosting lidera G-Mean (0,5342) e Recall-Ativo (0,30)
- A escolha se sustenta na natureza do entregável (priorização em faixas, não decisão binária) e na **captura em cobertura fixa**: 90% dos Ativos do hold-out alcançados abordando 35,6% da base, contra 49,6% do HistGradient Boosting e 52,5% da Regressão Logística
- O HistGradient Boosting fica registrado como alternativa para maximizar a detecção da classe minoritária, condicionada a calibração de probabilidades
- Nenhum modelo foi descartado por sobreajuste; o Δ CV-holdout é reportado, não usado como filtro

**Sobre os drivers de churn** (Gini, estabilidade entre partições e SHAP em três modelos):
- **Frequência de compras** é o driver mais robusto: 1º por Gini (23,19%), 1º em 30 de 30 partições e 1º nos dois ensembles por SHAP
- **Recência** é 2ª por SHAP nos três modelos, o achado mais robusto entre estimadores; no Top 3 por Gini em 22 de 30 partições
- **Diversidade de categorias** (`n_categories`) está no Top 3 por Gini em 29 de 30 partições, em empate técnico com a recência, mas é confirmada entre as quatro primeiras por dois dos três modelos (6ª no HistGradient Boosting). A leitura é correlacional, não causal
- **Satisfação declarada** (`avg_review_score`) tem importância secundária: o churn no Olist é, sobretudo, um problema de engajamento, não de qualidade
- A importância por permutação não é informativa neste regime amostral e não é usada como evidência

**Sobre a metodologia**:
- O filtro `frequency >= 2` é empiricamente validado: melhora o AUC em 0,2875 ponto e o MCC em 79%
- O undersampling aleatório com semente fixa preservou a distribuição da classe Churn (8 de 9 features com PSI < 0,10, nenhuma distorcida), com as ressalvas de potência do KS e de que a validação não alcança a expansão sintética da classe Ativo
- A divergência entre rankings de AUC e métricas robustas (Spearman entre 0,30 e 0,65, três vencedores distintos) confirma a necessidade do portfólio multi-métrica
- A concordância entre Gini e SHAP não é tratada como validação independente
- O protocolo de validação cruzada e de ajuste de limiar foi corrigido; a reamostragem fora do fold inflava a CV-AUC em até 0,3251, e escolher o limiar no teste inflava o Kappa em 0,0336

**Sobre a aplicação de negócio**:
- 1.178 clientes com score e segmento de risco atribuídos; taxa de churn observada estritamente monotônica entre as faixas (22,2% a 99,9%)
- Fora da amostra, a exclusão da faixa Crítico retém 9 dos 10 Ativos do hold-out em 39,0% da base, lift de 2,31x, estável em relação à base completa (2,34x)
- A diversidade de categorias como driver sugere cross-selling como mecanismo promissor de retenção, com a ressalva de que a evidência é correlacional

### Limitações reconhecidas

1. **Classe minoritária pequena**: 1.178 clientes após o filtro, com apenas 50 Ativos e 10 no hold-out. Cada Ativo capturado vale 10 pontos percentuais de recall, e diferenças de um cliente entre modelos não são conclusivas
2. **Regime amostral fora do estudado por De la Cruz Huayanay, Bazán e Russo (2025)**, que usaram 5.000 e 10.000 observações; o comportamento das métricas sob amostras pequenas é lacuna apontada pelos próprios autores
3. **MCC e Kappa moderados** (máximos de 0,35 e 0,29 entre os dez modelos): consequência matemática do hold-out com apenas 10 Ativos (LUQUE et al., 2019)
4. **Nenhum estimador de importância resolve sozinho a ordenação** neste regime; a permutação tem desvios de 44% a 93% das médias
5. **Label Encoding** trata categorias como grandeza ordenada; a validação quantitativa do oversampling gaussiano não foi feita
6. **Período histórico**: o dataset cobre 2016 a 2018 e pode não refletir dinâmicas atuais do mercado
7. **Variabilidade entre versões de bibliotecas**: mesmo com semente fixa, mudanças no scikit-learn podem produzir variações marginais nas métricas e na ordenação de features próximas

### Trabalhos futuros (conforme a Conclusão do TCC)

1. Features de sessão e comportamento de navegação
2. Modelos de churn em tempo real com processamento via streaming
3. Hiperparametrização via otimização bayesiana (Optuna)
4. Modelos de sobrevivência (Cox) para estimar o tempo até o churn
5. SHAP por instância no contrato de serving da API, para explicações personalizadas
6. Seleção de amostras típicas (clientes mais informativos) em substituição ao undersampling aleatório
7. Calibração de probabilidades (Platt scaling ou regressão isotônica) para os modelos de boosting
8. Exposição do modelo via protocolo MCP, para consulta de scores e explicações SHAP em linguagem natural
9. Codificação alternativa das variáveis categóricas (target encoding ou agrupamento de categorias raras)

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
| **Average Precision (AP)** | Área sob a curva Precision-Recall; sua linha de base é a prevalência da classe avaliada |
| **F1-Macro** | Média simples do F1 de cada classe |
| **MCC** | Matthews Correlation Coefficient, métrica simétrica para classificação binária |
| **G-Mean** | Raiz quadrada do produto entre Sensibilidade e Especificidade |
| **Kappa de Cohen** | Concordância corrigida pelo acerto esperado ao acaso |
| **SHAP** | SHapley Additive exPlanations, explicabilidade baseada em teoria dos jogos |
| **Participação relativa** | \|SHAP\| médio de uma feature dividido pela soma das onze do mesmo modelo; torna comparáveis modelos com saídas em escalas diferentes |
| **Importância por permutação** | Queda de desempenho ao embaralhar uma feature nos dados de avaliação; neste trabalho, não informativa por causa dos 10 Ativos do hold-out |
| **PSI** | Population Stability Index, mede deslocamento entre distribuições |
| **KS-test** | Kolmogorov-Smirnov, teste de igualdade entre duas distribuições |
| **Threshold (limiar)** | Valor de probabilidade acima do qual o cliente é classificado como Churn |
| **Corte de segmentação** | Limites nominais de 25%, 50% e 75% que definem as faixas de risco; não confundir com o limiar |
| **Cobertura** | Fração da base que a área de retenção consegue abordar (o orçamento) |
| **Captura** | Quantos Ativos reais estão dentro da cobertura (o retorno) |
| **Lift** | Densidade de Ativos no recorte dividida pela densidade na base inteira |
| **Stratified split** | Divisão treino/teste que preserva a proporção das classes |

---

## Referências mobilizadas

- **HUGHES (1994)**, Strategic Database Marketing: metodologia RFM (Seção 3)
- **KAUFMAN et al. (2012)**, Leakage in data mining: janelas temporais, ajuste do scaler e reamostragem dentro do fold (Seções 2, 6 e 7)
- **MATUSZELAŃSKI; KOPCZEWSKA (2022)**, Estudo Olist: filtro de frequência e horizonte de predição (Seções 2, 3, 4 e 4.1)
- **HE; GARCIA (2009)**, Imbalanced data: balanceamento e representatividade (Seções 6 e 8.5)
- **CHAWLA et al. (2002)**, SMOTE: comparação de estratégias (Seção 6)
- **HADDADI et al. (2024)**, Resampling para churn: estratégia adotada (Seção 6)
- **LEE (2000)**, Noisy replication: fundamento do oversampling gaussiano (Seção 6)
- **BISHOP (1995)**, Treino com ruído como regularização de Tikhonov (Seção 6)
- **BREIMAN (2001)**, Random Forest: modelo de produção e importância de Gini (Seções 7 e 8)
- **FRIEDMAN (2001)**, Gradient Boosting: algoritmos comparados (Seção 7)
- **VAFEIADIS et al. (2015)**, Comparação de algoritmos para churn: desenho do benchmark (Seção 7)
- **FAWCETT (2006)**, ROC analysis: ajuste de limiar (Seção 7)
- **CAWLEY; TALBOT (2010)**, Viés de seleção na avaliação: limiar derivado do treino (Seção 7)
- **HASTIE; TIBSHIRANI; FRIEDMAN (2009)**, Validação cruzada e comparação com hold-out (Seção 7)
- **DE LA CRUZ HUAYANAY; BAZÁN; RUSSO (2025)**, Métricas para classes desbalanceadas: protocolo multi-métrica e critério do limiar (Seções 7 e 7.1)
- **SAITO; REHMSMEIER (2015)**, Curva Precision-Recall em desbalanceamento (Seção 7)
- **CHICCO; JURMAN (2020)**, Vantagens do MCC (Seção 7)
- **LANDIS; KOCH (1977)**, Interpretação de concordância do Kappa (Seção 7)
- **LUQUE et al. (2019)**, Impacto do desbalanceamento em métricas: interpretação de MCC e Kappa moderados (Seção 7)
- **DAL POZZOLO et al. (2015)**, Calibração sob undersampling: deslocamento das probabilidades (Seções 7 e 9)
- **NICULESCU-MIZIL; CARUANA (2005)**, Calibração de probabilidades de boosting (Seções 7 e 9)
- **PRABADEVI; SHALINI; KAVITHA (2023)**, Regressão Logística como modelo de referência (Seção 7)
- **KUBAT; MATWIN (1997)**, One-sided selection e G-Mean (Seções 7 e 8.5)
- **SHAPLEY (1953)**, Teoria dos jogos: fundamento do SHAP (Seção 8.6)
- **LUNDBERG; LEE (2017)**, Unified approach to interpreting: SHAP (Seção 8.6 e comparação entre modelos)
- **LUNDBERG et al. (2020)**, TreeSHAP (Seção 8.6)
- **PEDREGOSA et al. (2011)**, Scikit-learn: variação entre versões (Seção 8 e limitações)
- **REICHHELD; SCHEFTER (2000)**, E-loyalty: fundamento econômico da segmentação (Seção 9)
- **KOTLER; KELLER (2016)**, Administração de Marketing: segmentação por risco e cross-selling (Seções 8 e 9)
- **HADDEN et al. (2007)**, Churn management: definição de churn implícito (Seção 4)
- **NESLIN et al. (2006)**, Defection detection: definição de churn implícito (Seção 4)
- **IMANI et al. (2025)**, Revisão sistemática: riscos metodológicos e trabalhos futuros (Seção 2 e Síntese)
- **MANZOOR et al. (2024)**, Recomendações para praticantes: features complementares ao RFM (Seção 3)
- **OLIST (2018)**, Dataset Olist no Kaggle: fonte de dados (Seção 1)

Citadas apenas no markdown do notebook, **fora da lista de referências do TCC**: MAAN; MAAN (2023) e EL ATTAR; EL-HAJJ (2026), ambas no bloco de introdução ao SHAP (Seção 8.6). Se o guia ou o notebook forem anexados à monografia, essas duas precisam entrar nas referências ou sair do texto.
