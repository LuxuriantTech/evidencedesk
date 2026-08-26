# ADR-0004 — Intégrité des évaluations et verrouillage local des holdouts

## Décision

Le runner doit exécuter chaque holdout une seule fois sous un nom verrouillé globalement. Le résultat
enregistre les hashes du manifeste et du corpus, le fingerprint du moteur, les paramètres, la graine
et le mode. Les verrous sont indépendants du répertoire de sortie afin qu'un second chemin ne
permette pas de rejouer le même holdout tant que le lock local persiste.

## Historique observé

- `holdout-2026.08.26.2-blind-extractive-local-v1.1-frozen` a produit un verdict `FAIL` : précision de citations 0,70, abstention 0,80 et F1 d’extraction 0,941176.
- `holdout-2026.08.26.3-blind-extractive-local-v1.2-frozen` a également produit un verdict `FAIL` : précision de citations 0,50, exactitude d’abstention 0,70 et F1 d’extraction 0,333333. Son manifeste SHA-256 est `06cc0c42744a87295154386e14273caa5d2c105f7ccbc402b90c2d63ebcdf9a9`, son corpus SHA-256 est `10a2f50b9aa27f9888364b4a18ea95867c50ce82b5ae196404a534b7b90c3273` et son fingerprint moteur est `78bba49f9798f43b6aab0a7c49930272d5abcd715c4b4e9e9b9d1e3939239e46`. Ils sont figés dans `artifacts/evaluations/holdout-2026.08.26.3-blind-extractive-local-v1.2-frozen.json` et son verrou global dans `artifacts/evaluations/locks/`.

Ces résultats ne doivent pas être présentés comme des objectifs atteints. Une amélioration après l’ouverture d’un holdout exige un nouveau corpus indépendant et une nouvelle exécution unique.

## Limite de preuve

Le lock local n'est ni signé, ni WORM, ni horodaté par un tiers. Il ne prouve pas l'absence de
consultation préalable, de tuning caché, de suppression du lock ou d'exécution par un autre script.
Les artefacts établissent une cohérence locale entre hashes, fingerprint, commit et résultat ; ils
ne doivent pas être décrits comme un scellement indépendant.

## Métriques et faiblesse connue

La métrique de citation vérifie l’alignement document/page puis accepte une relation de sous-chaîne
dans un sens ou l’autre. L’audit indépendant v3 a recalculé **0/8** citations sous égalité littérale,
contre 4/8 dans l’artefact historique. Le libellé « exact » est donc falsifié. Les évaluations futures
doivent imposer l’égalité littérale ou une règle normalisée préenregistrée et inclure une revue
humaine échantillonnée.

## Interdictions

Il est interdit de retuner l’algorithme contre un holdout déjà ouvert, de remplacer son artefact, de modifier ses questions ou de contourner le verrou par un autre chemin de sortie. Les résultats de développement ne remplacent jamais un résultat holdout.
