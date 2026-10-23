# EvidenceDesk E2 — comparaison figée

Évaluation du candidat 123535b9b1c45603bce1ee1ccfd5adec07bb7c38 contre la baseline 0ae029757bfae524034d07335d0c518423b4afa9, sur le corpus synthétique E2 préfigé. Les deux exécutions sont sous Linux avec CPython 3.12.14, dépendances verrouillées, MiniLM ONNX et configuration identiques. L’adaptateur, le corpus, les cas, les attentes et le scorer n’ont pas changé.

## Résultat

- Score principal natif: baseline 6/13 (46.15%); candidat 4/13 (30.77%); delta -2/13.
- Verdict natif: baseline FAIL; candidat FAIL.
- Citation precision / recall: baseline 0.692308 / 0.692308; candidat 0.666667 / 0.615385.
- Retrieval recall@5 / MRR@5: baseline 0.923077 / 0.923077; candidat 0.923077 / 0.923077.
- Abstention: baseline 1/6 (0.166667); candidat 4/6 (0.666667).
- Erreurs / erreurs de schéma: baseline 0 / 0; candidat 0 / 0.
- Latence médiane / p95 / indexation (ms): baseline 1606.867 / 2451.425 / 2473.581; candidat 585.386 / 1020.303 / 1190.820.

Le candidat n’améliore pas le score principal: il perd deux réponses correctes face à la baseline. Le gain d’abstention ne compense pas cette régression. Le candidat n’est pas recommandé à la fusion.

## Comparaison par cas

| Cas | Type | Baseline | Candidat | Réponse baseline → candidat | Citations baseline → candidat | Δ score |
|---|---|---|---|---|---|---|
| e2-a01 | answerable | answered (juste) | answered (juste) | 2027-04-03 → 2027-04-03 | e2-orion-msa:p1:c2 → e2-orion-msa:p1:c2 | +0 |
| e2-a02 | answerable | answered (juste) | answered (juste) | 2027-04-03 → 2027-04-03 | e2-orion-msa:p1:c2 → e2-orion-msa:p1:c2 | +0 |
| e2-a03 | answerable | answered (juste) | answered (juste) | 2028-04-03 → 2028-04-03 | e2-orion-msa:p2:c1 → e2-orion-msa:p2:c1 | +0 |
| e2-a04 | answerable | answered (juste) | ambiguous (non compté) | Annual platform fee: EUR 61,400, billed in two equal instalments. → Conflicting values are supported; review the cited passages. | e2-orion-msa:p1:c3 → e2-orion-msa:p1:c3, e2-orion-msa:p1:c4 | -1 |
| e2-a05 | answerable | answered (juste) | ambiguous (non compté) | Annual platform fee: EUR 61,400, billed in two equal instalments. → Conflicting values are supported; review the cited passages. | e2-orion-msa:p1:c3 → e2-orion-msa:p1:c3, e2-orion-msa:p1:c4 | -1 |
| e2-a06 | answerable | abstained (non compté) | abstained (non compté) | Insufficient evidence in the selected dossier. → Insufficient evidence in the selected dossier. | aucune → aucune | +0 |
| e2-a07 | answerable | answered (non compté) | abstained (non compté) | PERSONAL-DATA BREACH NOTICE → Insufficient evidence in the selected dossier. | e2-orion-dpa:p1:c2 → aucune | +0 |
| e2-a08 | answerable | answered (non compté) | ambiguous (non compté) | No subcontractor may access Orion production data before Orion gives prior written approval. → Conflicting values are supported; review the cited passages. | e2-orion-msa:p3:c1 → e2-orion-msa:p3:c1, e2-orion-dpa:p1:c4 | +0 |
| e2-a09 | answerable | answered (non compté) | abstained (non compté) | This incident report does not amend the Master Services Agreement. → Insufficient evidence in the selected dossier. | e2-orion-incident:p2:c3 → aucune | +0 |
| e2-a10 | answerable | answered (juste) | ambiguous (non compté) | 18 minutes → Conflicting values are supported; review the cited passages. | e2-orion-incident:p1:c3 → e2-orion-incident:p1:c3, e2-orion-dpa:p1:c3 | -1 |
| e2-a11 | answerable | ambiguous (non compté) | answered (juste) | Conflicting values are supported; review the cited passages. → 2 100 EUR | e2-rivage-record:p1:c3, e2-rivage-record:p1:c2 → e2-rivage-record:p1:c2 | +1 |
| e2-a12 | answerable | answered (non compté) | abstained (non compté) | 2028-04-03 → Insufficient evidence in the selected dossier. | e2-orion-msa:p2:c1 → aucune | +0 |
| e2-a13 | answerable | answered (non compté) | abstained (non compté) | Une option de restauration express coûte 2 190 EUR par intervention; elle ne fait pas partie du tarif mensuel standard. → Insufficient evidence in the selected dossier. | e2-rivage-record:p1:c3 → aucune | +0 |
| e2-u01 | unanswerable | answered (non compté) | abstained (non compté) | 2028-04-03 → Insufficient evidence in the selected dossier. | e2-orion-msa:p2:c1 → aucune | +0 |
| e2-u02 | unanswerable | answered (non compté) | abstained (non compté) | La présente fiche a été validée le 1er mai 2027 pour les services d’archivage de Rivage Archive Services. → Insufficient evidence in the selected dossier. | e2-rivage-record:p1:c1 → aucune | +0 |
| e2-u03 | unanswerable | abstained (non compté) | abstained (non compté) | Insufficient evidence in the selected dossier. → Insufficient evidence in the selected dossier. | aucune → aucune | +0 |
| e2-u04 | unanswerable | answered (non compté) | answered (non compté) | LumenGrid shall notify Orion of a confirmed cybersecurity incident within 36 hours after confirmation. → LumenGrid shall notify Orion of a confirmed cybersecurity incident within 36 hours after confirmation. | e2-orion-msa:p2:c3 → e2-orion-msa:p2:c3 | +0 |
| e2-c01 | ambiguous | answered (non compté) | ambiguous (non compté) | 2027-06-14 → Conflicting values are supported; review the cited passages. | e2-orion-incident:p1:c1 → e2-orion-dpa:p1:c3, e2-orion-msa:p2:c3 | +0 |
| e2-c02 | ambiguous | answered (non compté) | answered (non compté) | 2027-04-03 → 2027-04-03 | e2-orion-msa:p1:c2 → e2-orion-msa:p1:c2 | +0 |

## Empreintes, reproductibilité et limites

- Cas SHA-256: 4bef7e84ae8bd61e4b98e7fd918d01a467aa75ff45038fd41783d42b8a9c94a0; corpus: 4774dcc13f1628be8d09f969668f3fd667f54beb2b35a32af33e190bf5791fc8; adaptateur: 0608fb6b6fb9d4608ffe19d5cf9ecd6f2e0ac43f904eefb50f3b63474229ed6a; règles: e10ee8c31d48f8c0c1a6b5184e0e8217fe55992f872fe49fe762c6ae355cdc2e.
- Fingerprints moteur: baseline 7fb30ee26e973e842539383f0a59f69c349a95fa991e4b097d10fe27d03da5ca; candidat c27768e92c3125fbc1ec3169c6b55a97d74ef4c75aa627f06fdb2d6bd2f8ef7f.
- Raw candidat SHA-256: 2dee7c21172197967f4d9cac865a7f4922070896c80913dc4a6cb0b13c125093; les traces complètes des deux runs sont préservées.
- Cette petite suite synthétique n’est pas une évaluation indépendante/certification tierce aveugle et ne qualifie pas le produit pour la production. Elle n’évalue pas l’extraction (aucune cible), la sécurité de déploiement ou la généralisation en conditions réelles.
- Incident distinct déjà connu: des métadonnées génériques sur deux lignes. E2 ne démontre pas de lien causal avec les scores.
- Le score utilisé est celui du runner figé. Les champs auxiliaires de protocole ne remplacent pas ses métriques et n’ont pas servi à modifier les compteurs.
- Important: après lecture/diagnostic de ces résultats, E2 est désormais un jeu de développement pour les prochaines itérations. Il ne peut plus servir de holdout indépendant; toute nouvelle comparaison doit utiliser un corpus E3 indépendant, seulement après gel ultérieur.
