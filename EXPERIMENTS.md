# Experiments

Running log of every evaluated run, newest last. One row per run; the matching
`results/<id>.json` holds the full configuration and per-mixture metrics.

**Rules of the log**

1. Change one variable per experiment — otherwise a gain cannot be attributed.
2. Always report on a frozen benchmark split, never on a set tuned along the way.
3. A row is added by `src.eval.registry.save_results`, not by hand.

| Date | Experiment | Dataset | SI-SDRi (dB) | PESQ | STOI | Count acc. | Commit | Notes |
|---|---|---|---|---|---|---|---|---|
<!-- experiments:rows -->
