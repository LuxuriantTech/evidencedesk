# ADR-0006 — Moteur de réponse sourcée v3

## Statut

Décision figée localement avant le holdout v7. Le résultat de développement ne constitue pas une
validation de généralisation.

## Contexte

Le holdout v6 avait un Recall@5 de 1, mais seulement 2 réponses répondables correctes sur 25 et un
F1 d'extraction de 0,451613. Le passage pertinent était donc généralement récupéré ; la faiblesse
principale se situait dans la décision de répondre, l'alignement réponse–citation et la couverture
du schéma d'extraction. Les gold v2 à v6 ont été conservés immuables et n'ont pas servi au réglage
du moteur v3.

Le développement v3 emploie huit documents synthétiques et 48 cas répartis par familles de modèle
et de formulation entre calibration et sélection. La séparation n'est pas aléatoire. Le protocole,
les deux partitions et leurs SHA-256 sont enregistrés dans
`evals/configs/answer-v3-experiment-plan.json` et
`artifacts/evaluations/development_v3/frozen-runtime-v4-locked-final/summary.json`.

## Options bornées

Le plan préenregistré limitait l'expérience à trois stratégies et deux configurations par stratégie :

1. extracteur déterministe généralisé avec validation de preuve ;
2. NLI multilingue local `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, MIT, environ 300 M de
   paramètres, ONNX quantifié de 338 679 133 octets ;
3. sortie JSON contrainte avec `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, Apache-2.0, 490 M de paramètres,
   GGUF Q4_K_M de 491 400 032 octets.

Les identités, révisions, licences et empreintes de fichiers sont versionnées sous `infra/models/`.
Aucune API externe n'est appelée.

## Décision

La configuration `deterministic-v3-a` est retenue selon la règle préenregistrée : absence d'erreur
technique ou de schéma, stabilité sur trois exécutions, puis maximin des métriques de qualité, puis
p95, mémoire et simplicité. Ses seuils sont : support `0,48`, support partiel `0,38` et marge de
contradiction `0,08`.

Le moteur produit une décision explicite par candidat et une sortie finale distinguant preuve
supportée, support partiel, ambiguïté et abstention. La validation finale exige que le document et
la page existent, que l'extrait appartienne au chunk cité et que la réponse soit justifiable par cet
extrait. Une contradiction ne peut pas devenir une réponse certaine. L'extraction utilise le même
ensemble de passages et associe chaque valeur à sa preuve.

La configuration complète et les hashes sont dans
`evals/configs/answer-v3-frozen-v7.json`. L'empreinte moteur au moment de la sélection est
`fbff611f7bfa9dfdc81af6ca43925278b02400beead54ea3dd8cb0964cfed775`.

## Résultat de développement et limite

Sur la partition de sélection, la configuration retenue donne : exactitude répondable `0,9375`,
exactitude d'abstention `0,75`, précision/rappel des citations `1/1`, extraction P/R/F1
`0,953488/0,911111/0,931818`, Recall@5 `1`, MRR@5 `0,96875`, erreurs techniques et de schéma `0`.
Les décisions sont identiques sur trois exécutions. Sur la reproduction finale, la médiane agrégée
est `836,865 ms`, le p95 `1 026,154 ms` et le pic RSS `904 192 000` octets sur le poste documenté.

Le verdict de développement reste **FAIL**, car l'abstention `0,75` est sous l'objectif `0,85`.
La configuration est néanmoins le choix final imposé par la règle maximin ; aucun réglage n'a été
effectué après ouverture de la partition de sélection. Seul le holdout v7 indépendant et exécuté
une fois peut établir ou réfuter la généralisation.

## Conséquences

- Le mode par défaut devient `grounded-local-v3`, toujours sans clé et sans LLM génératif requis.
- La recherche hybride de l'ADR-0005 reste inchangée.
- Le NLI et Qwen sont des candidats évalués, pas des dépendances d'inférence du mode retenu.
- Les artefacts `frozen-runtime/`, `frozen-runtime-final/`, `frozen-runtime-v4-final/` et
  `frozen-runtime-v4-attested-final/` et `frozen-runtime-v4-freeze-final/` sont conservés comme
  incidents ou prédécesseurs et exclus de la preuve finale ; `frozen-runtime-v4-locked-final/` est
  la source de vérité.
- Un défaut de résolution des chemins relatifs du finalizer a été reproduit avant gel. L'essai
  incomplet et la comparaison alors invalidée restent archivés ; un test de régression a précédé
  la comparaison et la reproduction finales sous la nouvelle empreinte.
- Un échec du holdout v7 arrête l'expérimentation : aucun v8 ne doit être créé dans ce cycle.
