# Projeto B — Machine Learning, Ataques de Inferência e Análise de Trade-off em Privacidade Diferencial

## Visão Geral

Este repositório contém o **ambiente experimental de Machine Learning** do projeto **IC Privacidade**.  
Seu objetivo é avaliar, de forma **quantitativa, reproduzível e comparável**, como diferentes níveis de **Privacidade Diferencial (ε)** afetam simultaneamente:

- a **segurança dos dados**, medida por ataques de inferência;
- a **usabilidade dos dados**, medida por métricas estatísticas e performance de modelos;
- o **trade-off entre proteção e utilidade**.

O projeto consome **datasets previamente gerados** por um pipeline de engenharia de dados e **não aplica privacidade diferencial internamente**.

---

## Papel na Arquitetura do Projeto

O projeto completo é composto por três sistemas independentes:

1. **Projeto A — Sistema de RH (OLTP)**  
   - Geração de dados limpos e consistentes  
   - Simulação de ambiente corporativo real  

2. **Projeto Intermediário — DP Data Pipeline**  
   - Extração de dados do RH  
   - Aplicação de Privacidade Diferencial  
   - Versionamento de datasets  

3. **Projeto B — ML e Análise Experimental (este repositório)**  
   - Consumo dos datasets versionados  
   - Treinamento de modelos  
   - Execução de ataques de inferência  
   - Análise de utilidade e visualização de resultados  

Este repositório **não acessa diretamente o banco do sistema de RH**.

---

## Objetivo

Avaliar empiricamente como a variação do parâmetro de privacidade (ε) influencia:

- a taxa de sucesso de ataques de inferência e reidentificação;
- a performance e estabilidade de modelos de Machine Learning;
- a qualidade estatística dos dados para análise;
- a relação entre **segurança** e **usabilidade** em dados protegidos por Privacidade Diferencial.

---

## Escopo do Projeto

Este repositório é responsável por:

- Carregar datasets versionados gerados pelo pipeline;
- Treinar modelos de Machine Learning com e sem privacidade;
- Executar ataques de inferência sobre os modelos treinados;
- Calcular métricas de segurança e utilidade;
- Comparar resultados entre diferentes níveis de ε;
- Gerar visualizações e dashboards explicativos.

Este projeto **não**:
- gera dados primários;
- aplica mecanismos de Privacidade Diferencial;
- altera os datasets de origem.

---

## Modelos de Machine Learning

Os modelos são definidos de acordo com o contexto do sistema de RH, podendo incluir:

- regressão (ex.: previsão salarial);
- classificação (ex.: cargo ou setor);
- modelos supervisionados tradicionais.

Cada experimento inclui:
- um **baseline sem privacidade**;
- versões treinadas com datasets privatizados.

---

## Ataques de Inferência Avaliados

Os seguintes ataques são implementados e avaliados:

- **Membership Inference Attack**  
  Determina se um indivíduo específico fez parte do conjunto de treinamento.

- **Attribute Inference Attack**  
  Tenta inferir atributos sensíveis ocultos, como faixa salarial ou benefícios.

- **Model Inversion Attack**  
  Reconstrói características aproximadas de indivíduos a partir das saídas do modelo.

As taxas de sucesso são analisadas para diferentes valores de ε.

---

## Avaliação de Usabilidade dos Dados

A utilidade dos dados é avaliada por meio de:

- métricas de performance dos modelos (acurácia, erro, estabilidade);
- métricas estatísticas (distribuições, variância, correlação);
- comparação relativa entre datasets privatizados e o baseline.

Essas métricas permitem mensurar a **perda de utilidade causada pelo ruído**.

---

## Visualizações e Dashboards

O projeto inclui visualizações e dashboards com finalidade **exclusivamente explicativa**, utilizados para:

- sintetizar os resultados experimentais;
- visualizar o trade-off entre segurança e usabilidade;
- comparar métricas em função de ε;
- facilitar a interpretação dos resultados.

As visualizações **não influenciam decisões experimentais** e não fazem parte da geração das métricas.

---

## Estrutura Geral do Projeto (exemplo)

      project-b-ml-privacy/
    ├── datasets/ # referências às versões geradas pelo pipeline
    ├── models/ # definição e treino dos modelos
    ├── attacks/ # ataques de inferência
    ├── metrics/ # métricas de segurança e utilidade
    ├── analysis/ # análises estatísticas
    ├── dashboards/ # visualizações e BI explicativo
    ├── configs/ # parâmetros experimentais
    └── README.md


---

## Reprodutibilidade

Todos os experimentos são executados a partir de:

- uma versão explícita do dataset;
- valores conhecidos de ε;
- configurações controladas de modelos e métricas.

Isso garante reprodutibilidade, comparabilidade e isolamento dos resultados, mesmo com a evolução do sistema de origem.

---

## Motivação Acadêmica

Este projeto foi desenhado para:

- isolar a Privacidade Diferencial como variável experimental;
- avaliar simultaneamente segurança e utilidade;
- refletir cenários realistas de uso corporativo;
- produzir resultados sólidos para discussão acadêmica.

---

## Observações

- Os dados utilizados são simulados e não representam indivíduos reais.
- Este projeto é desenvolvido para fins acadêmicos e de pesquisa.
- Dashboards e visualizações têm caráter explicativo, não decisório.

---

## Licença

Uso acadêmico e educacional.

---

### Nota Final

Este repositório representa o **núcleo experimental do projeto**, onde o trade-off entre **Privacidade, Segurança e Usabilidade** é medido, comparado e explicado de forma controlada e reproduzível.





--- 

## Rascunho arquitetura pastas

      project-b-ml-privacy/
      │
      ├── datasets/
      │   └── README.md
      │   # apenas referência às pastas geradas pelo pipeline DP
      │
      ├── configs/
      │   ├── experiment.yaml        # YAML principal (ML + ataque)
      │   └── features.yaml          # (opcional) lista de features usadas
      │
      ├── data/
      │   ├── loader.py              # leitura dos CSVs + metadata
      │   └── splitter.py            # split treino/teste (seed fixa)
      │
      ├── models/
      │   ├── linear_regression.py   # treino do modelo
      │   └── evaluate.py            # métricas de utilidade (RMSE)
      │
      ├── attacks/
      │   └── membership/
      │       ├── attack.py          # execução do ataque
      │       └── evaluate.py        # métrica do ataque
      │
      ├── results/
      │   ├── raw/
      │   │   ├── utility.csv        # métricas de ML
      │   │   └── security.csv       # métricas de ataque
      │   │
      │   └── processed/
      │       └── tradeoff.csv       # dados prontos para análise
      │
      ├── analysis/
      │   ├── aggregate.py           # junta resultados por ε
      │   └── statistics.py          # médias, desvios, comparações
      │
      ├── dashboards/
      │   └── notebooks/
      │       └── tradeoff.ipynb     # visualização e storytelling
      │
      ├── run_experiment.py          # ponto de entrada único
      │
      └── README.md

