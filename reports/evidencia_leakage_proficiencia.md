# Evidência de leakage — proficiência determina o target

Consulta sobre `bronze_alfabetizacao.aeeb_ts_aluno` (apenas presentes):

| IN_ALFABETIZADO | n | min proficiência | max proficiência |
|---|---|---|---|
| 0 | 666,883 | 572.36 | 743.00 |
| 1 | 1,303,038 | 743.00 | 904.02 |

**Conclusão:** `IN_ALFABETIZADO = 1` ⇔ `VL_PROFICIENCIA_LP >= 743,0`.
Proficiência, respostas, gabaritos, caderno, blocos e peso do aluno são
**leakage total** e estão proibidos como features (ver `src/preprocessing/leakage.py`).