# Evaluation results (E1–E3)

All three evaluations use small synthetic corpora and frozen scoring inputs. None is a third-party blind certification or evidence of production quality. `FAIL` is the runner verdict, retained as recorded. Timings are environment-specific and should not be compared across different operating systems.

| Evaluation | Baseline | Candidate | Reading |
|---|---:|---:|---|
| E1, 20 cases (13 answerable) | 7/13 (53.85%), FAIL | 7/13 (53.85%), FAIL | No measured change; answer, status and citation outputs are unchanged across the 20 cases. Baseline ran on Windows and candidate on Linux, so timings are not comparable. |
| E2, 19 cases (13 answerable) | 6/13 (46.15%), FAIL | 4/13 (30.77%), FAIL | Candidate loses two answerable cases; abstention improves from 1/6 to 4/6, but does not offset the answer regression. |
| E3, 29 cases (17 answerable) | 10/17 (58.82%), FAIL | 7/17 (41.18%), FAIL | Candidate loses three answerable cases. Citation precision rises (0.909 to 1.000) while citation recall falls (0.588 to 0.412); abstention rises from 2/12 to 5/12. |

E1 and E2 are development evidence, not blind holdouts. E3 was authored as a synthetic independent suite but is no longer a fresh holdout after diagnostic review. The primary score is the frozen runner count `answerable_correct / answerable_total`; auxiliary interpretations do not replace it. E1 includes a secondary semantic note for one negation case, excluded from the score. E2 has no extraction golds. E3 has no extraction golds.

E3 incident record: the first baseline raw JSON was overwritten by the first candidate run, and the original baseline stdout was overwritten during recovery. Those first baseline bytes are unavailable. The comparison therefore uses the retained baseline recovery run (which matches the initially captured answer/citation counts and score; timings differ) and the exact first and only candidate run. The candidate raw file is preserved byte-for-byte; its stdout copy has only the absolute machine-local output path redacted. The reconstructed `NOT_RAW` transcript is excluded and must not be treated as a raw result. No second candidate inference was completed. No E4 dataset or run was found or included.
