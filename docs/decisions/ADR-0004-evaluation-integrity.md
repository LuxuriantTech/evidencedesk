# ADR-0004 — Intégrité des évaluations et verrouillage local des holdouts

## Décision

Chaque holdout est exécuté une seule fois sous un lock global local. Le résultat conserve hashes, fingerprint moteur, paramètres, graine et mode. Le lock est indépendant du répertoire de sortie.

## Historique

- v2 et v3 ont produit `FAIL`; l'audit détaillé est dans `docs/holdout_v2_v3_error_audit.md`.
- v4 a échoué au préflight : l'attestation ne contenait pas les SHA-256 requis. Aucun raw n'a été créé et le lock est conservé.
- v5 a échoué avant les cas par `KeyError: filename` dans `evaluate_manifest.build_chunks`. Aucun raw n'a été créé et le lock est conservé. Le validateur de dataset impose désormais un `filename` non vide avant l'exécution.
- v6 a produit une seule sortie raw (`033ac62953fe8e447443f69cb185080a17e72989f2f00b706686a6efefd05d03`) puis un recalcul séparé (`5d920dc073b5d839be3125188388743974f1434e5de26f229b304df5b46ea6ed`), tous deux `FAIL`.

## Limite de preuve

Un lock local n'est ni signé, ni WORM, ni horodaté par un tiers. Les artefacts établissent une cohérence locale, pas un scellement indépendant ni l'absence de consultation/tuning préalable.

## Interdictions

Il est interdit de retuner contre un holdout ouvert, de remplacer son artefact, de modifier ses cas ou de contourner le lock par un autre chemin. Une amélioration exige un nouveau corpus indépendant et une nouvelle exécution unique.
