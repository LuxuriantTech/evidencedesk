# EvidenceDesk — baseline sur la suite indépendante figée

Date d’exécution : 2026-10-23 (Europe/Brussels). Exécution locale, sans navigateur ni service payant.

## Portée et protocole

Cette exécution mesure uniquement la baseline publique du dépôt EvidenceDesk, HEAD `0ae029757bfae524034d07335d0c518423b4afa9`, dans le clone isolé `baseline-checkout`. Le clone est détaché et propre après exécution. L’empreinte calculée par la fonction du dépôt est `7fb30ee26e973e842539383f0a59f69c349a95fa991e4b097d10fe27d03da5ca`.

Les 20 cas et leurs attentes ont été gelés avant cette exécution dans `../datasets/evaluation_cases.json` (SHA-256 `85fc6e595f0780539bdf77d5950192f217d61f425a0efc719479f50e2be5b46b`). Le corpus synthétique fait 4 documents et SHA-256 `01d93dba59f0bd90293374c2a8f9aefeef807cb8953c83bcfd75c3943906631c`. Le protocole et les références de provenance sont dans `../PROTOCOL.md` et `../SHA256SUMS.json`. Répartition : 13 questions répondables, 5 sans réponse dans le corpus, 2 ambiguës. Aucune cible d’extraction n’est déclarée.

L’adaptateur commun est `../adapters/run_baseline.py`; il appelle le runner et le moteur v3 gelé du dépôt, avec la recherche hybride, le modèle MiniLM ONNX local réel, et la configuration `answer-v3-frozen-v7.json`. Le même fichier d’adaptateur doit être utilisé tel quel pour le candidat; seul `EVIDENCEDESK_REPO` change pour désigner son clone gelé. Le résultat brut complet, réponses, statuts, citations, traces de récupération, latences et métriques du runner sont dans `raw_results.json`.

L’empreinte de l’adaptateur, du moteur, des entrées et des fichiers du modèle est dans `ARTIFACTS_SHA256SUMS.json`. Le manifeste du modèle et tous ses fichiers ont été vérifiés contre leurs tailles et SHA-256 déclarés par le provider du dépôt. Le moteur rapporte l’identifiant `paraphrase-multilingual-minilm-l12-v2-onnx-q@faf4aa4225822f3bc6376869cb1164e8e3feedd0`; le poids ONNX est `model_optimized.onnx`, SHA-256 `634d0f66c29dc934c8fa72b8a4fe91dd4d420a22f1d82a241058d4316e659a99`. Calcul sur CPU, sans appel externe.

## Résultats exacts

| Mesure | Résultat |
|---|---:|
| **Score principal, compteur machine natif** | **7/13 (53,85 %)** |
| Observation sémantique secondaire pour a09 (non utilisée comme score) | 8/13 (61,54 %) |
| Citations correctes sur citations retournées | 8/13 (61,54 %) |
| Références attendues retrouvées exactement | 8/13 (61,54 %) |
| Récupération de la page attendue dans les 5 premières | 12/13 (92,31 %) |
| Réciproque du rang moyen, coupure 5 | 0,923077 |
| Abstention correcte, corpus insuffisant | 1/5 (20 %) |
| Ambiguïté correctement signalée | 0/2 (0 %) |
| Exactitude sur l’ensemble des 7 cas non répondables/ambigus | 1/7 (14,29 %) |
| Erreurs techniques / erreurs de schéma | 0/20 / 0/20 |
| Médiane / p95 par question | 2 823,100 / 4 224,604 ms |
| Indexation initiale | 1 346,080 ms |

Le score comparable principal est le compteur natif `answerable_correct=7/13`, selon le grader figé du runner. Les autres compteurs natifs sont `citation_correct=8/13`, `citation_expected_matched=8/13`, `abstention_correct=1/7`, `errors=0`, `schema_errors=0`; le verdict natif est `FAIL`. L’observation secondaire à 8/13 crédite a09 : la formulation citée exprime bien la négation demandée et la citation correspond exactement à la clause. Ce commentaire sémantique n’est pas ajouté au score principal et ne change ni gold ni aliases. Le F1 d’extraction natif vaut zéro parce que cette suite ne contient volontairement aucune cible d’extraction; il ne mesure donc pas une capacité d’extraction ici.

## Erreurs observées

| Cas | Résultat et problème |
|---|---|
| ind23-a01 | Donne la date Cobalt `2027-03-01` au lieu de la date d’effet Aster `2027-02-10`. |
| ind23-a02 | Répond avec la date de renouvellement `2028-02-10` à la question sur la prise d’effet Aster. |
| ind23-a10 | Répond sur la priorité de l’addendum au lieu de dire que le rapport d’incident ne modifie pas l’accord; citation DPA hors sujet. |
| ind23-a11 | Répond sur la priorité de l’addendum au lieu de choisir `EUR 42,800`; citation hors sujet. |
| ind23-a13 | Répond `2028-02-10` plutôt que dire que la date de renouvellement Cobalt n’est pas enregistrée. |
| ind23-u01 | Invente/importe la date de renouvellement Aster pour le contrat Cobalt, alors que le registre ne fournit pas cette date. |
| ind23-u02 | Transfère le tarif mensuel Cobalt vers l’accord Aster. |
| ind23-u03 | Retourne « not provided » comme réponse, au lieu de s’abstenir quand la question demande le numéro d’identification lui-même. |
| ind23-u04 | Statut `partially_supported` et citation d’une clause MSA, sans date de notification de l’incident. |
| ind23-c01 | S’abstient au lieu de signaler et citer les deux délais en conflit ainsi que l’absence de règle de priorité. |
| ind23-c02 | Choisit la date Cobalt pour « the supplier » sans résoudre l’identité du fournisseur, au lieu de signaler l’ambiguïté. |

Les cas justes sont ind23-a03, a04, a05, a06, a07, a08, a09, a12, et l’abstention ind23-u05. La liste complète, y compris les sorties exactes, est conservée dans les traces brutes.

## Limites

Il s’agit d’une suite indépendante synthétique de 20 cas, pas d’une évaluation tierce aveugle certifiée ni d’une preuve de généralisation à des contrats réels. Aucun jeu de comparaison candidat n’a été lancé. Les chiffres de latence sont propres à cette machine et à cette exécution. La métrique native d’extraction n’est pas interprétable pour cette suite puisque `extraction_targets` est vide. L’évaluation candidate attend le gel explicite du parent; aucun changement d’implémentation ni commit/push n’a été effectué ici.

