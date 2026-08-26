# Évaluation d'EvidenceDesk

Les évaluations locales mesurent recherche, réponse extractive, abstention et extraction sur corpus synthétiques versionnés. Elles ne mesurent ni un LLM, ni des données réelles, ni une charge API/SQL à grande échelle.

## Définitions figées des métriques

- **Précision des citations** : citations retournées dont document et page sont exacts et dont
  l'extrait normalisé est un sous-passage informatif de la preuve gold, divisées par toutes les
  citations retournées. Un extrait informatif contient au moins deux tokens alphanumériques.
- **Exactitude de citation par cas** : cas répondable au statut `answered`, valeur attendue reconnue
  par comparaison typée date/montant ou inclusion unidirectionnelle, et au moins une citation
  correcte, divisés par tous les cas répondables.
- **Exactitude d'abstention** : statut `abstained` pour les cas sans réponse ou adversariaux, et
  statut `ambiguous` pour les cas ambigus, divisé par ces cas non répondables.
- **Extraction P/R/F1** : micro-métriques sur chaque valeur gold du schéma ; une valeur prédite n'est
  vraie positive que si sa valeur typée et sa citation sont toutes deux correctes.
- **Recall@5 / MRR@5** : présence et rang réciproque de la première preuve gold parmi les cinq
  premiers passages récupérés, sur les questions répondables.
- **Latence** : temps mural par question du moteur en mémoire ; le p95 utilise le nearest-rank.
  L'indexation et le temps total sont enregistrés séparément.
- **Taux d'erreur** : exceptions moteur divisées par les cas tentés. **Coût** : coût déclaré des
  fournisseurs ; le mode local vaut 0 USD.

Les seuils gelés sont : précision et exactitude de citation au moins 0,90, exactitude d'abstention
au moins 0,85, F1 d'extraction au moins 0,90 et taux d'erreur nul. Les anciens holdouts v2/v3 ont
été audités après le gel et n'ont servi à aucun réglage de cette version.

## Développement sémantique v2 — itération 08

`artifacts/evaluations/development_v2_comparison/iteration-08-v6-final.json` compare quatre méthodes avec le même modèle ONNX local. Ces résultats de développement ne sont pas une validation holdout.

| Méthode | Citation P / cas | Abstention | Extraction F1 | Recall@5 / MRR@5 | Médiane / p95 ms | Verdict |
|---|---:|---:|---:|---:|---:|---|
| lexical | 1 / 1 | 1 | 1 | 0,833333 / 0,567778 | 38,435 / 48,034 | PASS développement |
| dense | 1 / 0,9 | 1 | 1 | 0,733333 / 0,502222 | 43,924 / 56,952 | PASS développement |
| hybride | 1 / 1 | 1 | 1 | 1 / 0,866667 | 41,247 / 50,054 | PASS développement |
| hybride + reranker | 1 / 1 | 1 | 1 | 0,966667 / 0,691667 | 46,380 / 59,398 | PASS développement |

La règle de sélection retient **hybride**. Le reranker est rejeté : aucune réponse répondable correcte supplémentaire et `reranker_retained: false`.

Le benchmark classe tous les chunks en mémoire. Le parcours API utilise SQL avec 20 candidats denses + 20 lexicaux ; son équivalence et sa performance à grande échelle ne sont pas prouvées ici.

## Holdouts et intégrité

v2 et v3 restent des `FAIL` historiques, détaillés dans `docs/holdout_v2_v3_error_audit.md`. Le lock local n'est ni WORM, ni signé, ni attesté par un tiers : il ne prouve pas l'absence de consultation préalable ou de suppression par un opérateur local.

- v4 : préflight échoué avant raw (`holdout attestation hashes are missing`) ; lock conservé.
- v5 : `KeyError: filename` dans `evaluate_manifest.build_chunks`, avant les cas et sans raw ; lock conservé. Le validateur exige désormais un `filename` non vide.
- v6 : une exécution raw unique a produit SHA-256 `033ac62953fe8e447443f69cb185080a17e72989f2f00b706686a6efefd05d03`; le recalcul séparé (`5d920dc073b5d839be3125188388743974f1434e5de26f229b304df5b46ea6ed`) confirme le verdict sans rouvrir le holdout.

### Protocole v4 figé et issue des tentatives

Le protocole v4, figé dans `evals/configs/semantic-v2-frozen.json` avant la génération, imposait :
graine `20260827`, 40 cas, au moins 25 répondables, 10 sans réponse et 5
ambigus/adversariaux, auteur indépendant, génération postérieure au gel moteur, gold immuable après
commit, une seule exécution et recalcul depuis le raw uniquement. Le jeu livré contenait exactement
25 cas répondables, 10 sans réponse, 3 ambigus, 2 adversariaux et 80 valeurs d'extraction. Son commit
dataset est `b5b7b36f58cfb6d36e6aea660d6c369509d1bbc0`, après le gel moteur
`39be32e011f10fed4453acbce6052b3a4e4fc85f`.

Cette tentative n'a cependant produit ni raw ni métrique : le runner a créé son lock puis rejeté
l'attestation, dont les hashes n'étaient pas sous la structure imbriquée attendue. Le recalcul v4 est
donc impossible et n'existe pas. Le même principe a été appliqué à v5, arrêté avant les cas par un
champ `filename` absent. Après correction structurelle du validateur, un nouveau v6 a été créé et
relu par deux agents indépendants du développement. Il a été soumis à un préflight complet sans
inférence avant la création du lock, puis exécuté une fois.

## Résultat holdout v6

Le raw v6 est `FAIL` : précision citation `0,4` (2/5), exactitude citation par cas `0,08` (2/25), abstention `0,8` (12/15), extraction P/R/F1 `1 / 0,291667 / 0,451613`, Recall@5 `1`, MRR@5 `0,9`, erreurs `0`, coût externe `0 USD`. Il s'abstient sur 20 des 25 cas répondables et traite strictement 0 des 3 cas ambigus comme ambigus. L'extraction est mesurée par champs : 35 vrais positifs sur 120 valeurs gold, 35 prédictions.

Temps runner local : indexation `2035,357 ms`, médiane `38,802 ms`, p95 `55,001 ms`, total `3651,221 ms`; pic RSS `698339328` octets. Ce ne sont pas des SLO de service. Les objectifs gelés ne sont pas atteints ; aucune optimisation ne doit être dérivée du corpus v6 ouvert.

Le diagnostic du raw explique le `FAIL` : la preuve gold est présente dans le top 5 pour 25/25
questions répondables, mais le moteur s'abstient sur 20 d'entre elles. Parmi les cinq réponses
retournées, deux sont entièrement correctes, deux ont la bonne valeur avec une mauvaise citation et
une a une valeur et une citation incorrectes. Les 10 cas sans réponse et les 2 cas adversariaux sont
correctement refusés, mais aucun des 3 cas ambigus n'obtient le statut strict attendu. Côté
extraction, les vrais positifs par champ sont : type `0/10`, date d'effet `5/10`, montants `20/20`,
obligations `0/20`, organisation `0/10`, renouvellement `10/10`, responsables `0/20` et risques
`0/20`. La faiblesse principale est donc la généralisation de la sélection/réponse et des patrons
d'extraction, pas la récupération top-5.

Recalcul vérifiable sans appeler le moteur ni rouvrir le corpus :

```bash
PYTHONPATH=apps/api:apps/worker:. uv run python -m evals.recalculate \
  --input artifacts/evaluations/holdout_v6/holdout-v6-raw.json \
  --output /tmp/holdout-v6-recalculated.json
sha256sum artifacts/evaluations/holdout_v6/holdout-v6-raw.json \
  artifacts/evaluations/holdout_v6/holdout-v6-recalculated.json
```

La copie synthétique `artifacts/evaluations/holdout-v6-summary.json` ne sert qu'à alimenter la page
de démonstration. Elle référence le SHA-256 du raw ; le raw et son recalcul restent les sources de
vérité métriques.

Dette de nommage : le runner partagé est toujours dans `evals/holdout_v4.py` et force
`schema_version: holdout-v4-raw-result-v2`. Le raw v6 porte donc ce label historique malgré son
`dataset_version` v6. Les provenance, hashes et métriques restent ceux de v6 ; une future version du
runner devra adopter un nom de schéma générique sans modifier cet artefact ouvert.

## Modèle et limites

Le modèle est `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` révision `faf4aa4225822f3bc6376869cb1164e8e3feedd0`, Apache-2.0, environ 118 M paramètres, 384 dimensions, maximum 512 tokens, pooling mean, CPU avec `fastembed==0.8.0` et `onnxruntime==1.29.0`. Téléchargement mesuré déclaré dans le manifest : `266906689` octets, dont `235052644` pour ONNX; pic développement : `725581824` octets (~725,6 MB). Les sept fichiers listés totalisent `266903238` octets, soit `3451` de moins que la mesure globale ; le manifest n'attribue pas cet écart. Le pooling mean est une convention fastembed, pas une validation indépendante d'optimalité. L'intégrité locale repose sur manifest et hashes, pas sur une signature de provenance.
