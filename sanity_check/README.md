Este modulo executa sanity checks para validar estabilidade dos modelos e do pipeline de MIA.
Ele nao interfere nos resultados finais dos experimentos de privacidade e existe apenas
para verificar comportamento basico antes da analise de trade-off.

Estrutura atual:

- `sanity_check/common.py`
  - Funcoes compartilhadas para executar modelos, dividir dataset e preparar sinais.
  - Reutiliza `build_preprocessor` e `compute_utility_metrics`.

- `sanity_check/model_checks.py`
  - Sanity checks do modelo (overfitting, underfitting, leakage, estabilidade).
  - Opera diretamente sobre saidas do modelo usando o mesmo preprocessor.

- `sanity_check/mia_checks.py`
  - Sanity checks do ataque MIA usando somente sinais precomputados
    (`train_abs_error`, `test_abs_error`).
  - Reutiliza `attacks.run_membership_inference_attack`.

- `sanity_check/sanity_model_validation.py`
  - Script para executar os sanity checks de modelo.

- `sanity_check/sanity_mia_validation.py`
  - Script para executar os sanity checks de MIA.

Como rodar:

- Sanity checks do modelo:
```bash
python sanity_check/sanity_model_validation.py
```

- Sanity checks do MIA:
```bash
python sanity_check/sanity_mia_validation.py
```
