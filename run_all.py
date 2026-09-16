"""Pipeline fim a fim — Fase 3 (requisito "pipeline reproduzível").

Executa: extract → features → Modelo B → ranking 2026 → clusters → interpret.

O **Modelo A** (`train_aluno`, ~33 min) fica **fora do padrão** (D14 — escopo
reduzido); inclua com ``--with-aluno``. Cada etapa é idempotente (cache de
parquets em ``data/``).

Uso::

    python run_all.py                 # ~5 min (sem Modelo A)
    python run_all.py --with-aluno    # inclui Modelo A completo (~38 min)
    python run_all.py --fast          # smoke (buscas curtas)
"""
from __future__ import annotations

import argparse
import logging
import time

log = logging.getLogger("run_all")


def _etapa(nome: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    log.info("=== %s ===", nome)
    fn(*args, **kwargs)
    log.info("--- %s concluída em %.1f s ---", nome, time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-aluno", action="store_true",
                    help="inclui o Modelo A completo (~33 min)")
    ap.add_argument("--fast", action="store_true", help="smoke (buscas curtas)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.data import build_features, extract_bq
    from src.evaluation import interpret
    from src.modeling import cluster_municipios, predict_2026

    _etapa("1/6 extract_bq", extract_bq.run_all)
    _etapa("2/6 build_features", build_features.run_all)

    if args.with_aluno:
        import subprocess
        import sys
        cmd = [sys.executable, "-m", "src.modeling.train_aluno"]
        if args.fast:
            cmd.append("--fast")
        _etapa("3/6 train_aluno (subprocesso)",
               lambda: subprocess.run(cmd, check=True))
    else:
        log.info("=== 3/6 train_aluno — PULADO (use --with-aluno; ~33 min, D14) ===")

    import subprocess
    import sys
    cmd_b = [sys.executable, "-m", "src.modeling.train_risco_meta"]
    if args.fast:
        cmd_b.append("--fast")
    _etapa("4/6 train_risco_meta", lambda: subprocess.run(cmd_b, check=True))

    _etapa("5/6 predict_2026 + clusters",
           lambda: (predict_2026.projetar_2026(), cluster_municipios.run()))
    _etapa("6/6 interpret", interpret.run)

    log.info("Pipeline concluído. Artefatos em reports/, models/, images/.")


if __name__ == "__main__":
    main()
