Este modulo executa testes de sanity check para verificar se os modelos apresentam:
- overfitting,
- underfitting,
- ou comportamento degenerado sob ruido elevado.

Ele **nao interfere nos resultados finais dos experimentos de privacidade** e existe apenas para validar a estabilidade basica dos modelos antes da analise de MIA e trade-off.

Arquivos adicionados:

- `sanity_check/sanity_mia_validation.py`
  - Objetivo: executar sanity checks avancados do Membership Inference Attack (MIA).
  - Valida se o ataque colapsa com rotulos aleatorios, se detecta diferencas extremas
    (train vs noise), se responde a aumento de overfitting e se mantem consistencia
    entre shadow e target (balanceamento e features).
  - Saida: lista de resultados por teste com status `ok/suspeito`.

- `sanity_check/sanity_model_validation.py`
  - Objetivo: executar sanity checks adicionais dos modelos (XGBoost e Random Forest).
  - Valida random label, overfitting controlado, underfitting controlado, data leakage,
    estabilidade entre seeds, sensibilidade ao tamanho do dataset e importancia de features.
  - Saida: lista de resultados por teste com status `ok/suspeito`.

Como rodar cada teste de sanidade:

- Sanity checks do modelo (XGBoost e Random Forest):
```bash
python sanity_check/sanity_model_validation.py
```

- Sanity checks do MIA:
```bash
python sanity_check/sanity_mia_validation.py
```
