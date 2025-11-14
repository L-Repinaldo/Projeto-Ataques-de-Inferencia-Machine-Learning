# Projeto-Ataques-de-Inferencia-Machine-Learning


🧬 Resumo do Projeto Machine Learning — Análise de Vazamento de Dados com Ataques de Inferência em ML e Privacidade Diferencial

O Projeto B é um ambiente de pesquisa experimental voltado a investigar como modelos de aprendizado de máquina podem vazar informações sensíveis quando treinados sobre dados protegidos por Privacidade Diferencial (DP).
Ele utiliza o mesmo banco de dados do Projeto A (Sistema de RH), porém com foco exclusivo em analisar riscos, simular ataques e medir a eficácia da proteção.

🎯 Objetivo Geral

Avaliar, de forma prática e reproduzível:

Quais tipos de ataques de inferência conseguem vazar informações sensíveis.

Como níveis diferentes de ruído (ε e δ) impactam a probabilidade de vazamento.

O trade-off entre privacidade e acurácia dos modelos.

A eficiência de mecanismos como Laplace, Gaussian e DP-SGD.

Quais atributos e padrões são mais suscetíveis a serem inferidos.

O projeto culmina em um artigo acadêmico comparando ataques, defesas e resultados.

🏗️ Base de Dados

O Projeto B acessa uma cópia ou segmento controlado do banco do Projeto A, incluindo:

funcionários (setor, faixa salarial, idade, cargo)

avaliações periódicas

benefícios utilizados

estrutura de setores e gerentes

Esses dados são ricos, sensíveis e ideais para simular cenários reais de vazamento.

🧠 Tipos de Ataques Implementados
1. Membership Inference Attack

Determina se um funcionário específico fez parte do conjunto de treinamento do modelo.

2. Attribute Inference Attack

Tenta prever atributos sensíveis ocultos, como:

faixa salarial

uso de determinados benefícios

nota de avaliação

setor de atuação

3. Model Inversion Attack

Reconstrói características aproximadas do indivíduo com base nas saídas do modelo.

Esses ataques são comparados com diferentes níveis de DP.

🔒 Mecanismos de Privacidade Avaliados

O Projeto B testa e compara:

Laplace Mechanism (para consultas agregadas)

Gaussian Mechanism

DP-SGD (treinamento com privacidade diferencial)

Perturbação de labels e features

Query-level vs. model-level DP

Cada mecanismo é analisado quanto a:

proteção efetiva

impacto na acurácia

resistência aos ataques

tempo de treinamento

📊 Métricas e Resultados

O sistema produz:

gráficos de vazamento por ε

curvas de ataque vs. defesa

impacto de DP na acurácia do modelo

estimativas de risco individual por atributo

tabelas comparativas entre mecanismos

Esses resultados formam a base do artigo.

🔬 Metodologia

Importar dados do Projeto A (cópia sanitizada).

Separar features sensíveis e não sensíveis.

Treinar modelos com e sem DP (ex.: regressão, random forests, redes simples).

Aplicar ataques de inferência.

Medir taxa de sucesso.

Analisar o comportamento sob diferentes valores de ε.

Gerar gráficos, relatórios e conclusões.

📌 Relação com o Projeto A

O Projeto A é o sistema “real protegido”.

O Projeto B é o ambiente de experimentação que tenta quebrar ou inferir informações do mesmo banco.

A comparação entre ruído aplicado no A e ataques no B permite gerar um artigo forte e bem fundamentado.
