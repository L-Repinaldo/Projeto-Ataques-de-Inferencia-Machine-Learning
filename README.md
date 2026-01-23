# Projeto B — Machine Learning, Ataques de Inferência e Análise de Trade-off em Privacidade Diferencial

## Visão Geral

Este repositório contém o núcleo experimental da pesquisa em Privacidade Diferencial, responsável por medir, comparar e explicar o trade-off entre segurança e usabilidade dos dados sob diferentes níveis de privacidade (ε).

O Projeto B utiliza Machine Learning como instrumento de medição de utilidade, ataques de inferência como instrumento de medição de risco, e IA como camada de síntese e explicação dos resultados. O foco não é otimizar modelos, mas tornar explícitos os efeitos da Privacidade Diferencial sobre o uso real dos dados.

Este repositório consome datasets previamente privatizados por um pipeline de engenharia de dados dedicado e não aplica mecanismos de DP internamente.

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
   - Consumo dos datasets versionados;  
   - Medição de utilidade via Machine Learning controlado;  
   - Execução de ataques de inferência;  
   - Comparação entre segurança e usabilidade;
   - Síntese e explicação dos resultados.

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

Este repositório é responsável por:

- Carregar datasets versionados gerados pelo pipeline;
- Treinar modelos de Machine Learning simples e interpretáveis;
- Executar ataques de inferência sobre os modelos;
- Calcular métricas de segurança e utilidade;
- Comparar resultados entre diferentes níveis de ε;
- Gerar gráficos e visualizações explicativas do trade-off.

Este projeto **não**:
- gera dados primários;
- aplica mecanismos de Privacidade Diferencial;
- altera os datasets de origem.

---

## Modelos de Machine Learning

O Machine Learning não é o foco do projeto, mas um instrumento prático de medição de usabilidade.

### Modelo Utilizado:
- Regressão Linear supervisionada

Justificativa:
- Modelo simples, interpretável e sensível a ruído;
- Facilita a observação direta do impacto da DP;
- Evita mascarar efeitos do ruído com técnicas robustas ou complexas.
  
Não há otimização, tuning agressivo ou comparação entre modelos.

---

## Métricas Utilizadas

- MAE (Erro Médio Absoluto)
  Mede quanto o modelo erra, em média.

- R² (Capacidade Explicativa)
   Indica quanto da estrutura dos dados ainda pode ser explicada.

- Estabilidade do Resultado
  Medida pela variação do MAE e do R² entre múltiplas execuções.

Essas métricas respondem perguntas simples:

- O dado ainda é útil?
- O aprendizado ainda é confiável?
- A estrutura dos dados foi preservada?
  
---

## Ataques de Inferência Avaliados

Os seguintes ataques são implementados e avaliados:

- **Membership Inference Attack**  
  Determina se um indivíduo específico fez parte do conjunto de treinamento.

- **Attribute Inference Attack**  
  Tenta inferir atributos sensíveis ocultos, como faixa salarial ou benefícios.

As taxas de sucesso são analisadas para diferentes valores de ε.

---

## Trade-off Segurança × Usabilidade

O trade-off é analisado a partir da simetria entre utilidade e risco:

- Dados mais úteis tendem a permitir maior vazamento;
- Dados mais protegidos tendem a perder capacidade de uso;
- O objetivo é identificar zonas intermediárias, onde algum nível de utilidade ainda é possível com risco controlado.

Nenhuma conclusão depende exclusivamente de métricas de ML ou de ataques isoladamente.

---

## Visualizações, Síntese e Uso de IA

As visualizações e a IA têm finalidade exclusivamente explicativa e comunicacional.

A IA não participa de nenhuma das etapas experimentais (geração de dados, treinamento de modelos, cálculo de métricas ou execução de ataques). Seu uso ocorre apenas após a obtenção dos resultados, com os seguintes objetivos:

- sintetizar os resultados experimentais obtidos;
- auxiliar na interpretação do trade-off entre segurança e usabilidade;
- organizar comparações entre diferentes valores de ε;
- apoiar a comunicação clara dos achados para o leitor.

Exemplos de gráficos utilizados:

- ε vs MAE (perda de utilidade);
- ε vs R² (colapso da estrutura);
- ε vs instabilidade do resultado;
- Utilidade vs sucesso do ataque (trade-off direto).

As visualizações e a IA não influenciam decisões experimentais e não fazem parte da geração das métricas.

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
- evitar viés de otimização de modelos;
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

Este repositório representa o ambiente experimental controlado onde o trade-off entre Privacidade Diferencial, Segurança e Usabilidade é medido, comparado e explicado, com Machine Learning atuando como instrumento, ataques como evidência de risco e IA como suporte à interpretação, mantendo o foco científico no fenômeno, não na ferramenta.
