# TCC em LaTeX — Análise Preditiva de Churn de Clientes utilizando Machine Learning

Projeto montado sobre o **Template MBA IA e Big Data LaTeX 3.2** (Pacote USPSC / ICMC-USP),
a partir do documento **TCC_Churn_v9** (Google Docs).

Autor: Fabiano Rodrigues Ugulino
Orientadora: Profa. Dra. Cibele M. Russo
Documento principal: `USPSC-modelo-ICMC-PORTUGUÊS.tex`

---

## 1. Como compilar

### No Overleaf (caminho mais simples)

1. Novo projeto > *Upload Project* > envie o zip inteiro.
2. Em *Menu > Settings*, defina:
   - **Compiler**: pdfLaTeX
   - **Main document**: `USPSC-modelo-ICMC-PORTUGUÊS.tex`
   - **TeX Live version**: 2023 ou superior
3. Compile. O Overleaf roda BibTeX automaticamente.

Observação: o Overleaf às vezes tem dificuldade com o acento no nome do arquivo
principal. Se der problema, renomeie para `USPSC-modelo-ICMC-PORTUGUES.tex`
(sem acento). Nenhum outro arquivo referencia esse nome.

### Localmente

A sequência precisa ser esta, e o `pdflatex` roda três vezes para resolver
sumário, listas e referências cruzadas:

```bash
pdflatex USPSC-modelo-ICMC-PORTUGUÊS.tex
bibtex   USPSC-modelo-ICMC-PORTUGUÊS
pdflatex USPSC-modelo-ICMC-PORTUGUÊS.tex
pdflatex USPSC-modelo-ICMC-PORTUGUÊS.tex
```

Pacotes TeX Live necessários (Ubuntu/Debian):

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-fonts-recommended texlive-lang-portuguese \
                 texlive-publishers texlive-science texlive-pictures lmodern
```

`texlive-publishers` é o que traz o abnTeX2, e `texlive-lang-portuguese`
o que traz o `brazil.ldf` do babel. Sem eles a compilação para logo no início.

### Estado atual da compilação

Última compilação: **83 páginas, 0 erros, 0 avisos de margem estourada
(overfull box), 0 referências ou citações indefinidas.**

---

## 2. Mapa dos arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `USPSC-modelo-ICMC-PORTUGUÊS.tex` | Documento principal: preâmbulo, pacotes, ordem dos elementos |
| `USPSC-pre-textual-ICMC.tex` | Título, autor, orientadora, ano, programa, textos da capa e folha de rosto |
| `USPSC-TA-PreTextual/USPSC-Resumo.tex` | Resumo + palavras-chave |
| `USPSC-TA-PreTextual/USPSC-Abstract.tex` | Abstract + keywords |
| `USPSC-TA-PreTextual/USPSC-Dedicatoria.tex` | Dedicatória |
| `USPSC-TA-PreTextual/USPSC-Agradecimentos.tex` | Agradecimentos |
| `USPSC-TA-PreTextual/USPSC-Epigrafe.tex` | Epígrafe (Deming) |
| `USPSC-TA-PreTextual/USPSC-AbreviaturasSiglas.tex` | Lista de siglas (25 entradas) |
| `USPSC-TA-PreTextual/USPSC-fichacatalografica.tex` | Ficha catalográfica (palavras-chave já ajustadas) |
| `USPSC-TA-Textual/USPSC-Cap1-Introducao.tex` | Cap. 1 — Introdução (1.1 a 1.5) |
| `USPSC-TA-Textual/USPSC-Cap2-Fundamentacao_Teorica.tex` | Cap. 2 — Referencial Teórico (2.1 a 2.7) |
| `USPSC-TA-Textual/USPSC-Cap3-Metodologia.tex` | Cap. 3 — Metodologia (Figs. 1-2, Tabs. 1-3, Quadro 1) |
| `USPSC-TA-Textual/USPSC-Cap4-Avaliacao_Experimental.tex` | Cap. 4 — Avaliação Experimental (Figs. 3-20, Tabs. 4-8, Quadro 2) |
| `USPSC-TA-Textual/USPSC-Cap5-Conclusao.tex` | Cap. 5 — Conclusão |
| `USPSC-bib/referencias-tcc.bib` | As 39 referências, em BibTeX ABNT |
| `USPSC-img/figuras/` | As 20 figuras (**hoje são placeholders**, ver seção 3) |
| `USPSC-classe/` | Classe e estilos do Pacote USPSC (não alterar, salvo o item 6.3) |

O arquivo `USPSC-bib/USPSC-modelo-references.bib` que veio no template foi
mantido intacto mas não é mais usado; a bibliografia aponta para
`referencias-tcc.bib`.

---

## 3. Substituição das figuras (pendência principal)

As 20 figuras hoje no projeto são **placeholders gerados automaticamente**: não
foi possível extrair as imagens embutidas no Google Doc. Cada arquivo mostra o
número da figura, o título e o nome do arquivo a ser substituído.

Para trocar, basta exportar a figura do notebook e salvar **com o mesmo nome**
em `USPSC-img/figuras/`. Nenhuma alteração nos `.tex` é necessária.

Recomendação: exporte em PDF ou PNG a 300 dpi
(`plt.savefig("fig01-pipeline.png", dpi=300, bbox_inches="tight")`). Se preferir
PDF, troque também a extensão nos comandos `\includegraphics` correspondentes.

| Nº | Arquivo esperado | Figura |
| --- | --- | --- |
| 1 | `fig01-pipeline.png` | Pipeline metodológico do estudo |
| 2 | `fig02-janela-temporal.png` | Estratégia de janela temporal dupla (anti-leakage) |
| 3 | `fig03-eda-recencia.png` | Análise exploratória de recência |
| 4 | `fig04-eda-frequencia.png` | Análise exploratória de frequência |
| 5 | `fig05-eda-review-score.png` | Análise exploratória de review score |
| 6 | `fig06-proporcao-churn-ativo.png` | Proporção Churn versus Ativo |
| 7 | `fig07-matrizes-confusao-a.png` | Matrizes de confusão: LogReg, Decision Tree, HistGB, AdaBoost, KNN |
| 8 | `fig08-matrizes-confusao-b.png` | Matrizes de confusão: Extra Trees, Gradient Boosting, Naive Bayes, SVM |
| 9 | `fig09-limiar-corte.png` | Critério de escolha do ponto de corte (Random Forest) |
| 10 | `fig10-curva-precision-recall.png` | Curva Precision-Recall por classe (Random Forest) |
| 11 | `fig11-importancia-gini.png` | Importância das features por impureza de Gini |
| 12 | `fig12-top5-gini.png` | Top-5 features por impureza de Gini |
| 13 | `fig13-shap-importancia-media.png` | Importância média \|SHAP\| das features |
| 14 | `fig14-shap-beeswarm.png` | Valores SHAP (beeswarm) no conjunto de teste |
| 15 | `fig15-shap-dependence-frequency.png` | Dependence plot de `frequency` |
| 16 | `fig16-shap-dependence-recency.png` | Dependence plot de `recency_days` |
| 17 | `fig17-shap-dependence-ncategories.png` | Dependence plot de `n_categories` |
| 18 | `fig18-shap-waterfall-ativo.png` | Decomposição SHAP: cliente representativo Ativo |
| 19 | `fig19-shap-waterfall-churn.png` | Decomposição SHAP: cliente representativo Churn |
| 20 | `fig20-distribuicao-scores.png` | Distribuição dos scores e limiares nos três modelos |

Depois de trocar as figuras, recompile e confira a paginação: a Lista de Figuras
e o Sumário se ajustam sozinhos, mas figuras com proporções muito diferentes das
atuais podem deslocar quebras de página.

---

## 4. Como citar no texto

A bibliografia usa o sistema autor-data do abnTeX2. Duas formas:

| Comando | Saída | Quando usar |
| --- | --- | --- |
| `\cite{breiman2001}` | (BREIMAN, 2001) | citação entre parênteses, no fim da frase |
| `\citeonline{breiman2001}` | Breiman (2001) | quando o autor faz parte da frase |

Várias de uma vez: `\cite{hadden2007,matuszelanski2022}` produz
(HADDEN et al., 2007; MATUSZELAŃSKI; KOPCZEWSKA, 2022).

Chaves disponíveis (39): `batista2004`, `bishop1995`, `breiman2001`,
`cawley2010`, `chawla2002`, `chicco2020`, `cohen1960`, `dalpozzolo2015`,
`delacruz2025`, `fawcett2006`, `friedman2001`, `haddadi2024`, `hadden2007`,
`hastie2009`, `he2009`, `hughes1994`, `imani2025`, `kaufman2012`, `kotler2016`,
`kubat1997`, `landis1977`, `lee2000`, `lundberg2017`, `lundberg2020`,
`luque2019`, `manzoor2024`, `matthews1975`, `matuszelanski2022`, `neslin2006`,
`niculescu2005`, `olist2018`, `pedregosa2011`, `prabadevi2023`, `reichheld2000`,
`saito2015`, `shapley1953`, `turban2018`, `ugulino2026`, `vafeiadis2015`.

Só aparecem nas REFERÊNCIAS as obras efetivamente citadas no texto. Todas as 39
estão citadas hoje.

---

## 5. Referências cruzadas

Seções, figuras, tabelas e quadros são referenciados por rótulo, não por número
fixo. Se você inserir uma seção nova no meio, a numeração se reajusta sozinha.

- Seções: `\ref{sec:benchmark}`, `\ref{sec:protocolo}`, `\ref{sec:interpretabilidade}` etc.
- Figuras: `\ref{fig:pipeline}`, `\ref{fig:shap-beeswarm}` etc.
- Tabelas: `\ref{tab:benchmark}`, `\ref{tab:segmentacao}` etc.
- Quadros: `\ref{qua:estrategias-balanceamento}`, `\ref{qua:tres-modelos}`
- Objetivo específico (4): `\ref{obj:importancia}`

---

## 6. Decisões que valem a sua conferência

### 6.1 Caixa alta nas citações entre parênteses

O template vem configurado conforme a **NBR 10520:2023**, que produz
"(Manzoor et al., 2024)". Seu texto no Google Docs usa caixa alta,
"(MANZOOR et al., 2024)". Para reproduzir o seu documento, foi adicionada a
opção `abnt-cite-style=AUTHORYEAR` no preâmbulo (linha do
`\usepackage[...]{abntex2cite}`).

Para voltar ao padrão do template, troque `AUTHORYEAR` por `AuthorYEAR` ou
remova a opção. Vale confirmar com a orientadora ou com a biblioteca do ICMC
qual das duas normas eles esperam.

### 6.2 Sobrenomes compostos nas referências

"De la Cruz Huayanay" e "Dal Pozzolo" precisaram ficar entre chaves no `.bib`,
porque sem isso o BibTeX quebra o sobrenome e imprime "HUAYANAY, A. de la C." e
"POZZOLO, A. D.". Com as chaves, o nome sai correto no texto corrido
(que é onde aparecem mais vezes), mas na lista de REFERÊNCIAS esses dois nomes
saem em caixa mista, e não em caixa alta como os demais.

Se preferir o inverso (caixa alta nas REFERÊNCIAS, ao custo de caixa alta
também no texto corrido), troque em `referencias-tcc.bib`:

```
{De la Cruz Huayanay}  ->  {DE LA CRUZ HUAYANAY}
{Dal Pozzolo}          ->  {DAL POZZOLO}
```

### 6.3 Erro de sintaxe no arquivo de estilo do template

O arquivo `USPSC-classe/abntex2-alf-USPSC.bst`, como distribuído, tem um erro de
sintaxe BibTeX na **linha 1293** (`{"{"t` — falta um espaço entre o literal e a
variável, numa alteração datada de 22/11/2023 no próprio arquivo). O BibTeX
reclama, mas gera o `.bbl` completo, então **não bloqueia a compilação** e não
afeta este trabalho.

Se quiser silenciar o aviso, o conserto é trocar `{"{"t` por `{"{" t` nessa
linha. Vale reportar à equipe do Pacote USPSC.

### 6.4 Itálico em termos estrangeiros

O texto foi transposto **fiel ao original**: termos como *churn*, *pipeline*,
*e-commerce*, *ensemble* e *hold-out* estão em redondo, como no Google Docs.
Se a orientadora pedir itálico nos estrangeirismos, é uma passada de revisão
sobre os cinco capítulos.

### 6.5 Código Cutter

O campo `\cutter{ }` está em branco em `USPSC-pre-textual-ICMC.tex`, como no seu
documento. A Biblioteca Prof. Achille Bassi atribui o código definitivo; quando
receber, preencha ali. Há uma linha comentada logo abaixo com uma sugestão
(`U26a`), caso queira algo provisório.

### 6.6 Folha de aprovação e ficha definitiva

Ambas estão como o template prevê para a versão original (folha de aprovação
substituída por página em branco). Depois da defesa:

- Salve a folha assinada como PDF e troque, no documento principal, a linha
  `\includepdf{USPSC-TA-PreTextual/USPSC-PaginaEmBranco.pdf}` (a que fica logo
  após o bloco da folha de aprovação) por
  `\includepdf{USPSC-TA-PreTextual/USPSC-folhadeaprovacao.pdf}`.
- Se a biblioteca enviar a ficha catalográfica em PDF, comente
  `\include{USPSC-TA-PreTextual/USPSC-fichacatalografica}` e descomente
  `\includepdf{USPSC-TA-PreTextual/USPSC-fichacatalografica.pdf}`.
- Para a versão revisada, troque `\notafolharosto{Vers\~ao original}` por
  `\notafolharosto{Vers\~ao revisada}` em `USPSC-pre-textual-ICMC.tex`.

---

## 7. Correções feitas em relação ao Google Docs

Duas coisas foram ajustadas na transposição:

1. **Legenda da Figura 13**: no original estava
   "Importância média |SHAP| das features (Random**Tabela 3 - Robustez das
   estratégias de balanceamento em trinta divisões estratificadas** Forest)",
   com um trecho da Tabela 3 colado no meio por acidente. Ficou
   "Importância média |SHAP| das features (Random Forest)", como consta na
   Lista de Figuras.
2. **Fim da Seção 4.7**: havia um espaço antes do ponto final em
   "alcançar 90% dos clientes com propensão real ao retorno ."

Nenhum número, resultado ou argumento do texto foi alterado.

---

## 8. Detalhes técnicos, caso precise mexer

- **`USPSC-pre-textual-ICMC.tex` usa escapes LaTeX** (`\^e`, `\c{c}`, `\~a`) em
  vez de acentos em UTF-8. Isso é obrigatório: a classe carrega esse arquivo
  antes do `\usepackage[utf8]{inputenc}`, e acentos diretos ali quebram a
  compilação. Todos os outros arquivos usam UTF-8 normalmente.
- **Quadro 2** não cabe em uma página, então foi feito com `longtable` e legenda
  via `\quadrocaption`, macro criada no preâmbulo com `\newfixedcaption`
  (recurso nativo do memoir). Ele quebra entre páginas repetindo o cabeçalho e
  entra normalmente na Lista de Quadros.
- **Tabela 4** (13 colunas) usa `\scriptsize`, `\tabcolsep` reduzido e
  cabeçalhos empilhados com `\shortstack` para caber na largura da página.
- Tabelas e quadros usam `\SingleSpacing` internamente, como pede a ABNT, apesar
  do corpo do texto estar em espaço 1,5.
- O **índice remissivo** foi desativado no documento principal, já que o trabalho
  não tem entradas `\index{}`. Para reativar, descomente o
  `\include{USPSC-TA-PosTextual/USPSC-IndicesRemissivos}` no fim do arquivo.
- Os **apêndices e anexos** seguem comentados, já que a v9 incorporou as antigas
  figuras A1-A3 ao corpo do texto.
