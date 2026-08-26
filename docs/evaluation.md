# Évaluation d'EvidenceDesk

## Question mesurée

L'évaluation demande si le mode `extractive-local` retrouve une preuve, répond ou s'abstient, et
extrait des champs simples sur des documents synthétiques inconnus du moteur gelé. Elle ne mesure
ni la qualité d'un LLM, ni une adoption utilisateur, ni la latence de la pile Docker complète.

## Protocole déclaré et contrôles locaux

- 40 cas par version combinée : 21 développement et 19 holdout ;
- total par version : 25 répondables, 10 sans réponse, 5 ambigus/adversariaux ;
- paramètres, graine, corpus, citations attendues et cibles d'extraction versionnés ;
- validation structurelle sans importer le moteur ;
- commit/fingerprint moteur gelé avant l'ouverture ;
- lock global créé de façon atomique, indépendant du répertoire de sortie ;
- le runner refuse localement une seconde exécution tant que son lock persiste ;
- après un FAIL, correction sur développement puis nouveau holdout indépendant ;
- v2 et v3 déclarés interdits à toute optimisation future.

Les attestations `engine_executed: false` sont utiles mais auto-déclaratives. Elles ne constituent pas
une preuve externe d'absence de consultation, de tuning ou de réexécution. Le lock n'est ni signé ni
WORM et peut être supprimé par un opérateur local. Les hashes et l'ordre commit → lock rendent les
fichiers présents cohérents et réduisent le risque sans établir un scellement indépendant.

## Définitions

- **Précision des citations du runner historique** : citations dont document et page correspondent
  et dont l'extrait attendu est une sous-chaîne de l'extrait retourné, ou inversement, divisées par
  toutes les citations retournées.
- **Exactitude par cas répondable** : réponse au statut `answered`, contenant la réponse attendue et
  au moins une citation acceptée, divisée par les cas répondables.
- **Exactitude d'abstention** : `abstained`, ou `ambiguous` pour un cas ambigu, divisé par les cas
  non répondables et ambigus/adversariaux.
- **Extraction** : micro-précision, micro-rappel et micro-F1 sur les valeurs ciblées, avec citation
  attendue par valeur lorsque le manifeste la fournit.
- **Latence** : temps mural de ranking/réponse en mémoire par cas ; chargement, API, DB et worker
  exclus. p95 par nearest rank.
- **Taux d'erreur** : exceptions moteur divisées par les cas tentés.
- **Coût** : somme déclarée par les fournisseurs ; 0 USD pour `extractive-local`.

Les objectifs scellés étaient : citations ≥ 90 %, abstention ≥ 85 %, extraction F1 ≥ 90 % et taux
d'erreur nul. Aucune p-value n'est produite : Benjamini-Hochberg/FDR n'est pas applicable.

## Résultats enregistrés

| Artefact | Cas | Citations | Cas répondables | Abstention | Extraction P/R/F1 | Médiane / p95 | Erreurs | Coût | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| développement v1.2 | 21 | 16/16 = 100 % | 16/16 = 100 % | 5/5 = 100 % | 9/9/9 = 100/100/100 % | 0,338 / 0,533 ms | 0/21 | 0 USD | PASS développement |
| holdout v2 | 19 | 7/10 = 70 % | 7/9 = 77,78 % | 8/10 = 80 % | 8/8/9 = 100/88,89/94,12 % | 1,448 / 1,976 ms | 0/19 | 0 USD | FAIL |
| holdout v3 | 19 | 4/8 = 50 % | 3/9 = 33,33 % | 7/10 = 70 % | 3/3/15 = 100/20/33,33 % | 0,532 / 1,021 ms | 0/19 | 0 USD | FAIL |

Fichiers :

- `artifacts/evaluations/development-2026.08.26.1-extractive-local-v1.2-frozen.json` ;
- `artifacts/evaluations/holdout-2026.08.26.2-blind-extractive-local-v1.1-frozen.json` ;
- `artifacts/evaluations/holdout-2026.08.26.3-blind-extractive-local-v1.2-frozen.json` ;
- `artifacts/evaluations/locks/` pour les verrous globaux.

## Audit adversarial v3

Les SHA manifest `06cc0c…cdf9a9`, corpus `10a2f5…c3273` et fingerprint moteur
`78bba4…39e46` concordent entre attestation, lock et artefact. Le commit moteur `d8b5dab` précède
le lock du 2026-08-26 à 18:53:01 UTC. Cette chronologie et les hashes ne prouvent pas que le dataset
n'a jamais été consulté auparavant ni qu'aucun autre runner n'a été utilisé.

Le libellé historique « extrait exact » est falsifié : `_citation_matches` accepte les sous-chaînes
dans les deux sens. Les quatre citations v3 comptées correctes englobent ou ponctuent l'extrait gold ;
aucune n'est littéralement égale. Recalcul strict sur les sorties archivées : 0/8. Ce chiffre ne
remplace pas rétroactivement l'artefact gelé ; il en documente la faiblesse.

La F1 extraction porte seulement sur 15 valeurs ciblées en v3, pas sur chaque champ du schéma. Le
runner reconstruit les chunks depuis le manifeste et appelle le moteur directement ; le vrai
parcours Docker est validé séparément par Playwright, sans prétendre que ce test E2E est un benchmark
de qualité documentaire.

## Interprétation

Le mode déterministe fonctionne sur le corpus de développement mais généralise mal à des domaines
et formulations nouvelles. Les résultats sont compatibles avec une méthode surtout lexicale et des
règles d'extraction liées au vocabulaire fournisseur. Le prochain travail empirique doit partir
d'une métrique de citation à égalité stricte, d'un extracteur plus général et d'un nouveau holdout v4
créé seulement après gel. Les v2 et v3 ne doivent pas être réutilisés pour optimiser.
