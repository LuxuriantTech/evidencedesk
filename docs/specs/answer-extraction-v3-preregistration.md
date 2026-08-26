# Pré-enregistrement — réponse et extraction v3

Date : 2026-08-27. Ce document et `evals/configs/answer-v3-experiment-plan.json` sont écrits avant
le premier benchmark sur development v3.

## Hypothèse et frontière

La recherche hybride reste figée. L'hypothèse testée est qu'une représentation explicite de la
preuve, une extraction de valeur séparée et une validation finale support/contradiction réduisent
les abstentions incorrectes sans accepter les questions sans preuve. Aucun contenu des holdouts
v2 à v6 n'est une entrée de développement ou une source de règle.

Le corpus v3 possède deux partitions par groupes, et non par tirage aléatoire : quatre familles de
gabarits/formulations pour la calibration, quatre autres pour la sélection. Après ouverture des
résultats de sélection, aucune configuration ne sera retouchée.

## Candidats retenus avant résultats

1. `deterministic-evidence-v3` : extraction déterministe généralisée, scoring sémantique local et
   validation structurale. Licence MIT du projet, aucun poids supplémentaire.
2. `multilingual-nli-v3` : mDeBERTa v3 base MNLI/XNLI, ONNX quantifié, pour valider entailment et
   contradiction. Révision `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`, licence MIT, environ
   0,3 milliard de paramètres, fichier ONNX de 338 679 133 octets.
3. `qwen-constrained-json-v3` : Qwen2.5 0.5B Instruct Q4_K_M, sortie contrainte par schéma JSON.
   Révision `9217f5db79a29953eb74d5343926648285ec7e67`, licence Apache-2.0,
   0,49 milliard de paramètres, GGUF de 491 400 032 octets.

Les licences, révisions et tailles ont été vérifiées le 2026-08-27 dans les métadonnées des dépôts
Hugging Face officiels des modèles. `deepset/tinyroberta-squad2` a été examiné puis rejeté avant
présélection : anglais uniquement et licence CC-BY-4.0, moins adapté au corpus bilingue et à une
intégration simple.

## Budget et configurations

- trois stratégies maximum ;
- deux configurations maximum par stratégie, toutes déclarées dans le JSON pré-enregistré ;
- trois répétitions déterministes par configuration ;
- aucun nouveau candidat après lecture des résultats ;
- coût API externe autorisé : zéro.

La sélection rejette d'abord toute erreur technique, erreur de schéma ou instabilité. Elle maximise
ensuite le minimum entre exactitude répondable, abstention, précision/rappel citation et F1
extraction sur la partition de sélection. Les égalités sont départagées par p95, RSS puis simplicité.

## Structure de sortie obligatoire

Chaque décision doit contenir `answerable`, `answer`, `confidence`, `supporting_document`,
`supporting_page`, `supporting_excerpt`, `ambiguity_reason` et `extracted_fields`. La validation
finale doit confirmer que document/page existent, que l'extrait est un sous-passage réel, que la
valeur est supportée, que les contradictions deviennent `ambiguous` et que le schéma est strict.
