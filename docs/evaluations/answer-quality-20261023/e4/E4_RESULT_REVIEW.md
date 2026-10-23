# Revue indépendante des résultats E4

**Périmètre :** comparaison en lecture seule des reçus, sorties natives, questions et documents sources des exécutions E4 baseline et candidate. Je n’ai consulté ni les valeurs attendues des cas, ni le code du scorer, ni les alias, et je n’ai relancé aucune évaluation.

## Intégrité des deux exécutions

Les reçus baseline et candidate indiquent `completed`, code retour 0 et les commits attendus (`0ae0297` et `ffeba12`). Les SHA-256 des fichiers `raw_results.json` correspondent aux reçus de fin. Les deux sorties portent les mêmes hashes de suite et de corpus; les hashes des entrées des deux exécutions correspondent au manifeste figé. Le résultat candidat provient de `split=development`, sur un corpus synthétique interne de 28 cas, et ne représente pas un test externe.

## Score natif

| Mesure | Baseline | Candidate | Lecture |
|---|---:|---:|---|
| Exactitude des 18 questions répondables | 2/18 (11,1 %) | 2/18 (11,1 %) | Inchangée |
| Précision des citations | 0,20 | 0,40 | 2 citations correctes dans les deux runs; citations rendues 10 → 5 |
| Rappel des citations | 2/18 (11,1 %) | 2/18 (11,1 %) | Inchangé |
| Exactitude d’abstention | 3/10 (30 %) | 5/10 (50 %) | Amélioration, mais encore une erreur sur deux cas non répondables/ambigus |
| Rappel de récupération à 5 | 18/18 | 18/18 | Stable |
| Erreurs / erreurs de schéma | 0 / 0 | 0 / 0 | Aucun défaut d’exécution détecté |

Le score natif du candidat est explicitement **`FAIL`**. La hausse de précision des citations vient de citations moins nombreuses, pas de nouvelles citations correctes; le rappel et l’exactitude de réponse n’augmentent pas. La précision relative s’améliore, mais l’exactitude ne s’améliore pas.

## Ce que montrent les cas et les sources

- Le candidat s’abstient à juste titre sur plusieurs faits présents en source, par exemple les questions du plan/permit, le modèle CTD-204, le pic de turbidité, le lien de SR-901 avec SB-031, le pH, le service order et l’heure de remplacement. Cela réduit les mauvaises réponses mais limite fortement l’utilité et explique en partie le score d’exactitude faible.
- Le cas `e4-a10` reste une réponse fausse et confiante : la question demande la salinité à 10:15, mais les deux sorties répondent avec la deuxième lecture à 10:40.
- Le cas `e4-c04` reste une réponse unique alors que le registre source contient deux observations actuelles non réconciliées, Q-3 et Q-4, toutes deux marquées inondées. Le candidat ne détecte donc pas cette ambiguïté.
- Sur `e4-a14`, la baseline cite le nitrate de SB-013 bien que le texte source précise que ce lot est distinct; la candidate s’abstient. C’est un gain de prudence concret, sans gain d’exactitude native globale.
- Sur `e4-a06`, la baseline confond l’accès au site BR-4 avec le point d’accès du permis, qui est North Gate. La candidate s’abstient au lieu de répéter cette confusion.
- Le champ d’extraction `responsible_people` contient dans les deux résultats `SiteCode BR-`, valeur tronquée et mal classée. L’évaluation des extractions est vide (zéro cible et zéro prédiction comptées), donc E4 ne valide pas la qualité d’extraction.

## Avis de publication

La candidate est plus sélective et s’abstient davantage, mais elle n’a pas démontré une meilleure exactitude. Le résultat natif est `FAIL`, l’exactitude reste 2/18, le rappel citationnel reste 2/18 et des erreurs de sélection de valeur ainsi qu’une ambiguïté manquée subsistent. Je ne recommande pas de la présenter comme améliorée, qualifiée pour la production ou prête à être mise en service. Un compte rendu interne d’expérience peut être partagé uniquement en affichant clairement le `FAIL`, le 2/18 inchangé, la nature synthétique et interne du jeu, le gain limité de précision des citations par réduction de leur nombre, et les risques ci-dessus. Le protocole E4 précise que cette évaluation n’autorise pas à publier, pousser ou fusionner.
