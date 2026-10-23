# EvidenceDesk — comparaison baseline / candidat

Date : 2026-10-23. Le candidat gelé `42d6ef9608a2487e0b08700e68e561108a7b54b2` a été évalué dans un clone Linux detached isolé. Le parent est `0ae029757bfae524034d07335d0c518423b4afa9`.

## Comparabilité

- Suite et gold inchangés : SHA-256 `85fc6e595f0780539bdf77d5950192f217d61f425a0efc719479f50e2be5b46b`.
- Corpus inchangé : SHA-256 `01d93dba59f0bd90293374c2a8f9aefeef807cb8953c83bcfd75c3943906631c`.
- Adaptateur identique : SHA-256 `0608fb6b6fb9d4608ffe19d5cf9ecd6f2e0ac43f904eefb50f3b63474229ed6a`.
- Règles et seuils : `../baseline-run/COMPARISON_RULES_FROZEN.json`. Les cinq fichiers d’évaluation du candidat sont byte-identiques aux hashes figés.
- Modèle MiniLM ONNX réel : `paraphrase-multilingual-minilm-l12-v2-onnx-q@faf4aa4225822f3bc6376869cb1164e8e3feedd0`; manifeste SHA-256 `69b870fae9bae6060e43aace617bf8cc17d3e200bfbaf06172c3219f79eb49ec`. Même cache vérifié.
- Baseline: Windows CPython 3.12.13; candidat: Linux CPython 3.12.14 avec dépendances verrouillées. Les latences ne sont donc pas comparables.
- Fingerprint baseline `7fb30ee26e973e842539383f0a59f69c349a95fa991e4b097d10fe27d03da5ca`; candidat `7c69d88f1891685e434712bc8cb8da384cff2fe8363d22192c4587ddf01664a5`.

## Résultats machine figés

| Métrique | Baseline | Candidat |
|---|---:|---:|
| Score principal du runner | 7/13 (53,85 %) | 7/13 (53,85 %) |
| Précision / rappel des citations | 0.615385 / 0.615385 | 0.615385 / 0.615385 |
| Retrieval recall@5 | 0.923077 | 0.923077 |
| Abstention correcte, 7 non répondables/ambiguës | 1/7 | 1/7 |
| Erreurs techniques / schéma | 0/0 | 0/0 |
| Verdict | FAIL | FAIL |

**Le candidat n’améliore pas les réponses mesurées.** Le score principal reste 7/13 et le verdict `FAIL`. Réponses modifiées : 0/20; statuts modifiés : 0/20; citations modifiées : 0/20. Les mêmes six cas répondables échouent : ind23-a01, ind23-a02, ind23-a09, ind23-a10, ind23-a11, ind23-a13. Les mêmes six cas non répondables ou ambigus échouent : ind23-u01, ind23-u02, ind23-u03, ind23-u04, ind23-c01, ind23-c02. Aucun cas ne passe de faux à vrai ni de vrai à faux. Les sorties détaillées sont dans `case_comparison.json` et `raw_results.json`.

L’observation manuelle secondaire sur ind23-a09 (la citation semble respecter la polarité négative) reste hors score pour les deux versions. Les écarts de temps s’expliquent par les OS et environnements Python différents; ils ne prouvent pas un gain du moteur.

## Traces

- Baseline SHA-256 : `01a857af7e624f08cbdfe32cce95d4819497bf82b40b226fc4ed2ebe043fd377`.
- Candidat SHA-256 : `7ee645e82137d666daacd6268280fce01f0429f0b08ac27fcc076200e2843e6f`.
- Stdout candidat : `runner_stdout.txt`.
