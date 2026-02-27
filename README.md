# Machine Learning, Ataques de Inferência e Análise de Trade-off em Privacidade Diferencial

### Este repositório acompanha um estudo experimental sobre Privacidade Diferencial e Membership Inference Attacks, com foco em análise de trade-off e reprodutibilidade.

## Visão Geral

Este repositório contém o núcleo experimental da pesquisa em Privacidade Diferencial, responsável por medir, comparar e explicar o trade-off entre segurança e usabilidade dos dados sob diferentes níveis de privacidade (ε).

O projeto utiliza Machine Learning como instrumento de medição de utilidade e ataques de inferência como instrumento de medição de risco. O foco não é otimizar modelos, mas tornar explícitos os efeitos da Privacidade Diferencial sobre o uso real dos dados.

Todos os experimentos consomem datasets previamente privatizados pelo pipeline de engenharia de dados (Projeto Intermediário) e **não aplicam mecanismos de DP internamente**.

---

## Papel na Arquitetura do Projeto

O projeto completo é composto por três sistemas independentes:

   1. **Projeto Sistema de RH**  
      - Simulação de um ambiente corporativo real;  
      - Geração de dados limpos, consistentes e sensíveis;  
      - Aplicação de regras de negócio.
   
   2. **Projeto DP Data Pipeline**  
      - Extração de dados do RH;  
      - Aplicação de Privacidade Diferencial com diferentes valores de ε;  
      - Versionamento de datasets;  
      - Geração de metadados experimentais.
   
   3. **Projeto ML e Análise Experimental (este repositório)**  
      - Carrega datasets versionados;  
      - Treina modelos de Machine Learning simples e interpretáveis;  
      - Executa ataques de inferência;  
      - Calcula métricas de utilidade e risco;  
      - Gera gráficos e tabelas de síntese para análise do trade-off.

Este repositório **não acessa diretamente o banco do sistema de RH**.

---

## Objetivo

Avaliar empiricamente como a variação do parâmetro de privacidade (ε) influencia:
   
   - Segurança dos dados, medida por ataques de inferência;  
   - Usabilidade dos dados, medida por métricas simples de Machine Learning;  
   - Estabilidade e confiabilidade do aprendizado;  
   - O trade-off entre proteção e utilidade em cenários realistas.

O objetivo central é demonstrar o trade-off, não maximizar performance nem propor novos modelos.

---

## Escopo do Projeto

Responsabilidades deste repositório:
   
   - Carregar datasets versionados gerados pelo pipeline;  
   - Treinar modelos de Machine Learning simples e interpretáveis;  
   - Executar ataques de inferência sobre os modelos;  
   - Calcular métricas de segurança e utilidade;  
   - Comparar resultados entre diferentes níveis de ε;  
   - Gerar gráficos e tabelas de síntese do trade-off.

Este projeto **não**:
   
   - Gera dados primários;  
   - Aplica mecanismos de Privacidade Diferencial;  
   - Altera os datasets de origem.

---

## Modelos de Machine Learning

O Machine Learning é usado apenas como instrumento de medição de usabilidade:

Modelos implementados:

   - XGBoost  
   - Random Forest

Justificativa:
   
   - Modelos relativamente simples e sensíveis a ruído;
   - Estruturas baseadas em árvores facilitam análise de estabilidade sob ruído;
   - Evitam mascarar efeitos da DP com arquiteturas altamente regularizadas ou profundas.  

Não há tuning agressivo, otimização ou comparação competitiva entre modelos.

---

## Métricas Utilizadas:

   - **Métricas de Utilidade:**
      - ***MAE (Erro Médio Absoluto):*** avalia a precisão de modelos de regressão.
      - ***RMSE (Raiz erro quadrático médio):*** avalia o desempenho de modelos de regressão, calculando a raiz quadrada da média dos erros ao quadrado entre os valores previstos e reais
  
   - **Métricas de Segurança:**
      - **attack_acc:** acurácia global do classificador de ataque.
      - **member_acc:** taxa de acerto do ataque em amostras que realmente pertencem ao treino (TP / (TP + FN)).
      - **non_member_acc:** taxa de acerto do ataque em amostras que não pertencem ao treino (TN / (TN + FP)).
      - **precision:** proporção de amostras preditas como member que realmente são members.
      - **recall:** capacidade do ataque de identificar corretamente members reais.
   
Essas métricas respondem perguntas simples:
   
   - O dado ainda é útil?  
   - O aprendizado ainda é confiável?  
   - A estrutura dos dados foi preservada?

---

## Ataques de Inferência Avaliados
   
   - **Membership Inference Attack (MIA):** determina se um indivíduo fez parte do conjunto de treinamento.
     
As taxas de sucesso são analisadas para diferentes valores de ε.

---

## Trade-off Segurança × Usabilidade

O trade-off é analisado a partir da simetria entre utilidade e risco:
   
   - Dados mais úteis tendem a permitir maior vazamento;  
   - Dados mais protegidos tendem a perder capacidade de uso;  
   - O objetivo é identificar zonas intermediárias onde algum nível de utilidade ainda é possível com risco controlado.

Nenhuma conclusão depende exclusivamente de métricas de ML ou de ataques isoladamente.

---

## Visualizações e Tabelas de Síntese

Este repositório gera visualizações e tabelas que permitem comunicar os resultados de forma clara:
   
   - **Plots:**
      - ***Utilidade:***  
           - Tabelas e gráficos das métricas de utilidade

      - ***Segurança:***
           - Tabela expositiva das métricas referentes à segurança
           
      - ***Trade-Off:***
           - Utilidade vs sucesso do ataque (trade-off direto)  
      
      - **Tabelas de síntese:**  
           - Relacionam cada modelo com usabilidade e segurança por nível de ε  

As visualizações têm caráter explicativo e **não influenciam decisões experimentais**.

---

## Estrutura Geral do Projeto
      
      project-b-ml-privacy/
      ├── analysis/ 
      ├── attacks/ 
      ├── data/ 
      ├── experiments/ 
      ├── metrics/ 
      ├── model/ 
      ├── plots/ 
      ├── preprocessing/
      ├── sanity_check/
      ├──main.py
      ├──config.py
      └── README.md



---

## Reprodutibilidade

Todos os experimentos são executados a partir de:
   
   - datasets explicitamente versionados;  
   - valores conhecidos de ε;  
   - configurações controladas de modelos e métricas;  
   - seleção do dataset ativo via `config.py`.

A versão do dataset utilizada nos experimentos é definida por `DATASET_VERSION` em `config.py`, garantindo comparabilidade e reprodutibilidade entre execuções.

Este repositório **não gera dados** e **não aplica DP internamente** — apenas consome versões previamente privatizadas pelo pipeline.

---

## Como rodar

   1. Defina a versão do dataset em `config.py`:
         ```python
         DATASET_VERSION = "v-2026-02-07_15-53-36"
      
   2. Garanta que os datasets estejam em:
         ```python
         data/datasets/<DATASET_VERSION>/
         
   3. Execute:
         ```python
         python main.py
         
   Os datasets devem ter sido previamente gerados e privatizados pelo pipeline de DP.
   
---

## Motivação Acadêmica

Este projeto foi desenhado para:

   - Isolar a Privacidade Diferencial como variável experimental;  
   - Avaliar simultaneamente segurança e usabilidade;  
   - Evitar viés de otimização de modelos;  
   - Produzir resultados sólidos para discussão acadêmica.

---

## Observações
   
   - Os dados utilizados são simulados e não representam indivíduos reais.  
   - Este projeto é desenvolvido para fins acadêmicos e de pesquisa.  
   - Visualizações e tabelas têm caráter explicativo, não decisório.

---

## Licença

Uso acadêmico e educacional.

---

### Nota Final

Este repositório representa o ambiente experimental controlado onde o trade-off entre Privacidade Diferencial, Segurança e Usabilidade é medido, comparado e explicado, com Machine Learning atuando como instrumento, ataques como evidência de risco, e visualizações e tabelas para síntese, mantendo o foco científico no fenômeno, não na ferramenta.


