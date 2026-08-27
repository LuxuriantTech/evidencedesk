# Analyse postérieure du holdout v7

## Périmètre et intégrité

Cette analyse décrit le seul lock/raw v7 enregistré dans l'historique local : elle ne modifie ni le
moteur, ni le dataset, ni les seuils. Le raw est `artifacts/evaluations/holdout_v7/raw.json` (SHA-256
`55a79af0d38ea1d7742763406e29dc96416f7c3cb158aa9a90bbbbd16650c1c1`) ; son recalcul séparé
est `recalculated.json` (SHA-256
`fd9e2864cd6b2aa4701cc1e2b979d49cd02e4fea76fbe8727271a45c9585fee3`). Le lock créé avant
inférence a le SHA-256 `03b455f0485d04afce46cf5798ce537786169a34d9b51787a372e121e01ac909`.

Le corpus, les cas et l'attestation ont respectivement les SHA-256
`32f0d0bda0abdc94cf2477dd640f82a7b37b7b2062bdc512d5480d5ac267d574`,
`2ab6c710a2eed039ae8749f151eb5894f875dd1411b69fcec9b106571580b1a1` et
`b60b2a74e69ca52a67117d610d15a5d7b95c5d131b478a5fa74fbe00d8ed9021`. Le recalcul vérifie
25/25 preuves dans le top-5, MRR@5 `0,94`, les latences enregistrées, l'indexation, le temps total
et le pic RSS. Le recalcul réagrège les décisions enregistrées et ne réévalue pas leur vérité
sémantique. Ces fichiers ne constituent pas un scellement WORM ou tiers et ne peuvent pas exclure
l'existence passée d'un run local ensuite supprimé.

## Résultat brut et recalculé

| Mesure | Brut | Recalcul |
|---|---:|---:|
| Précision / rappel citations | 0,80 / 0,48 | 0,80 / 0,48 |
| Exactitude répondable | 0,36 (9/25) | 0,36 (9/25) |
| Abstention | 0,80 (12/15) | 0,80 (12/15) |
| Extraction P / R / F1 | 0,659091 / 0,349398 / 0,456693 | identique |
| Recall@5 / MRR@5 | 1,00 / 0,94 | 1,00 / 0,94 |
| Erreur / schéma | 0 / 0 | 0 / 0 |
| Médiane / p95 | 756,641 / 1150,454 ms | contrôlé cohérent |

Les compteurs officiels de citations sont calculés sur les 25 cas répondables : 12 correctes sur 15
citations retournées, 12 preuves gold retrouvées sur 25. Le raw contient 20 citations sur tous les
types de cas. Le verdict brut et recalculé est `FAIL`. Le temps total est `36288,053 ms`, l'indexation
`3778,661 ms`, le pic RSS `700862464` octets et le coût externe `0 USD`.

Le taux d'erreur technique est `0` : il mesure les exceptions moteur, pas la justesse sémantique. Il
coexiste avec 19 décisions incorrectes sur les 40 cas selon le contrat de chaque type de question.

## Erreurs de réponse et d'abstention

La récupération n'est pas la cause principale : chaque question répondable a une preuve gold dans
les cinq premiers passages. En revanche, 11/25 questions répondables sont `abstained`; les deux
réponses `partially_supported` sont `v7-a21` et `v7-a23`; `v7-a04` choisit une organisation au lieu
du relecteur; `v7-a12` confond date d'effet et renouvellement. Seules 9/25 réponses répondables ont
valeur et citation exactes.

Les dix questions sans réponse sont refusées. Parmi les cas difficiles, seul `v7-m03` est marqué
ambigu : `v7-m01` et `v7-m02` reçoivent une réponse certaine malgré des preuves concurrentes.
`v7-x01` est refusé, mais `v7-x02` restitue une instruction présente dans le document. Il s'agit d'un
échec de sécurité du moteur de réponse, pas d'une simple divergence de formulation.

`v7-a16` est un défaut de mesure identifié après ouverture : la réponse cite correctement
« Le plafond de remorquage est de 31 600 EUR », mais le comparateur de montant prend l'horodatage
`06:23` comme premier nombre et marque `answer_match=false`. Il ne faut pas modifier le score v7;
le contre-factuel favorable ne ferait passer l'exactitude répondable qu'à 10/25, loin de 90 %.

## Erreurs d'extraction

| Champ | Vrais positifs | Prédits | Gold | Lecture |
|---|---:|---:|---:|---|
| type de document | 0 | 0 | 10 | jamais émis |
| organisation | 0 | 0 | 10 | jamais émise |
| date d'effet | 1 | 1 | 10 | formats variés mal généralisés |
| montants | 7 | 7 | 11 | couverture partielle |
| obligations | 6 | 8 | 10 | deux sur-prédictions |
| renouvellement | 5 | 6 | 9 | contradiction et formats variés |
| responsables | 4 | 10 | 11 | organisations confondues avec personnes |
| risques | 6 | 12 | 12 | granularité et extraits courts mal alignés |

Les erreurs dominantes sont donc l'absence de deux champs, les formes linguistiques non couvertes,
la confusion rôle/organisation, et la gestion insuffisante des contradictions. Certaines valeurs de
risque courtes sont sémantiquement proches mais échouent la comparaison textuelle stricte; cette
limite est distincte des nombreuses valeurs réellement absentes.

## Décision

Le moteur, les seuils et le protocole sont gelés pour ce cycle. Aucun ajustement sur v7, aucune
seconde exécution et aucun v8 ne sont autorisés ici. Le statut est `HONEST_NEGATIVE`. Une future
itération devra repartir d'un développement séparé, améliorer d'abord la résistance aux injections,
la décision d'answerability et le schéma d'extraction, puis utiliser un nouveau holdout indépendant.
