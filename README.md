# Projeto B — Machine Learning, Ataques de Inferência e Análise de Trade-off em Privacidade Diferencial

## Visão Geral

Este repositório contém o núcleo experimental da pesquisa em Privacidade Diferencial, responsável por medir, comparar e explicar o trade-off entre segurança e usabilidade dos dados sob diferentes níveis de privacidade (ε).

O Projeto B utiliza Machine Learning como instrumento de medição de utilidade e ataques de inferência como instrumento de medição de risco. O foco não é otimizar modelos, mas tornar explícitos os efeitos da Privacidade Diferencial sobre o uso real dos dados.

Todos os experimentos consomem datasets previamente privatizados pelo pipeline de engenharia de dados (Projeto Intermediário) e **não aplicam mecanismos de DP internamente**.

---

## Papel na Arquitetura do Projeto

O projeto completo é composto por três sistemas independentes:

   1. **Projeto A — Sistema de RH (OLTP)**  
      - Simulação de um ambiente corporativo real;  
      - Geração de dados limpos, consistentes e sensíveis;  
      - Aplicação de regras de negócio.
   
   2. **Projeto Intermediário — DP Data Pipeline**  
      - Extração de dados do RH;  
      - Aplicação de Privacidade Diferencial com diferentes valores de ε;  
      - Versionamento de datasets;  
      - Geração de metadados experimentais.
   
   3. **Projeto B — ML e Análise Experimental (este repositório)**  
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
   
   - Regressão Linear  
   - Elastic Net  
   - XGBoost  
   - Random Forest

Justificativa:
   
   - Modelos simples, interpretáveis e sensíveis a ruído;  
   - Facilitam a observação direta do impacto da DP;  
   - Evitam mascarar efeitos do ruído com técnicas complexas.  

Não há tuning agressivo, otimização ou comparação competitiva entre modelos.

---

## Métricas Utilizadas
   
   - **MAE (Erro Médio Absoluto):** mede quanto o modelo erra, em média.  
   - **R² (Capacidade Explicativa):** indica quanto da estrutura dos dados ainda pode ser explicada.  
   - **Estabilidade:** mede a variação do MAE e R² entre múltiplas execuções.  

Essas métricas respondem perguntas simples:
   
   - O dado ainda é útil?  
   - O aprendizado ainda é confiável?  
   - A estrutura dos dados foi preservada?

---

## Ataques de Inferência Avaliados
   
   - **Membership Inference Attack (MIA):** determina se um indivíduo fez parte do conjunto de treinamento.  
   - **Attribute Inference Attack (AIA):** tenta inferir atributos sensíveis ocultos, como faixa salarial ou benefícios.

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
   
   - **Gráficos:**  
     - ε vs MAE (perda de utilidade)  
     - ε vs R² (colapso da estrutura)  
     - ε vs instabilidade  
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
      ├──main.py
      └── README.md



---

## Reprodutibilidade

Todos os experimentos são executados a partir de:
   
   - datasets explicitamente versionados;  
   - valores conhecidos de ε;  
   - configurações controladas de modelos e métricas.

Isso garante comparabilidade e isolamento dos resultados.

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

