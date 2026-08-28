# Audit forensique quarantainé des holdouts historiques v2 et v3

**Statut : FAIL historique confirmé.** Ce document est un diagnostic des
artefacts déjà ouverts ; il ne rejoue aucun holdout, ne propose aucun réglage
et ne reproduit ni questions, ni réponses attendues, ni passages du corpus.
Les identifiants ci-dessous servent uniquement à relier une erreur à son
artefact archivé.

## Périmètre et intégrité

- v2 : artefact `artifacts/evaluations/holdout-2026.08.26.2-blind-extractive-local-v1.1-frozen.json`, verrou associé et moteur au commit `079118f15d52` (attesté par le verrou).
- v3 : artefact `artifacts/evaluations/holdout-2026.08.26.3-blind-extractive-local-v1.2-frozen.json`, verrou associé et moteur au commit `d8b5dab7ed000adc52fc6cf3279f0dfcfe4f0696` (commit de l'attestation et fingerprint de l'artefact).
- Vérification SHA-256 des corpus/manifests : v2 `10/10` fichiers attestés ; v3 `2/2` fichiers générés attestés ; aucun écart.
- Vérification d'immuabilité avant création de ce rapport : `git diff --quiet HEAD -- artifacts/evaluations datasets/blind_holdout_v2 datasets/blind_holdout_v3` a retourné `0`. Les artefacts, corpus, manifests et verrous historiques sont donc identiques à `HEAD`.
- Aucun appel au runner ni au moteur n'a été exécuté. Les constats proviennent de la comparaison des résultats archivés avec leurs manifests et de la lecture des sources aux commits figés.

## Recalcul indépendant des agrégats archivés

Sortie du calcul à partir des compteurs contenus dans les deux JSON archivés :

```text
v2: citation=7/10=0.700000; answerable=7/9=0.777778; abstention=8/10=0.800000; extraction=8/8/9 -> P=1.000000 R=0.888889 F1=0.941176; archived=0.700000/0.777778/0.800000/0.941176
v3: citation=4/8=0.500000; answerable=3/9=0.333333; abstention=7/10=0.700000; extraction=3/3/15 -> P=1.000000 R=0.200000 F1=0.333333; archived=0.500000/0.333333/0.700000/0.333333
```

Les valeurs recalculées sont égales aux valeurs archivées. Les deux résultats
restent donc `FAIL` : v2 manque les seuils citation et abstention ; v3 manque
les trois seuils.

## Règles de classement utilisées

- **Récupération/citation** : le passage sélectionné ne satisfait pas la
  citation attendue, et la réponse extractive dérive de ce passage.
- **Abstention incorrecte** : un cas répondable est refusé, ou un cas qui doit
  être refusé/ambigu reçoit une réponse.
- **Extraction trop rigide** : une valeur attendue est absente de l'extraction,
  sans valeur ou citation concurrente dans l'artefact.
- **Formulation/normalisation de vérité attendue** : réservé à un désaccord
  purement de forme. Aucun cas ne relève exclusivement de cette catégorie.
- **Combinaison** : au moins deux catégories précédentes s'appliquent au même
  cas. Les totaux par catégorie se chevauchent donc volontairement.

## Cas de questions v2

| Identifiant | Résultat attendu | Résultat observé | Classement | Cause démontrée dans le chemin d'exécution |
|---|---|---|---|---|
| `hold2-a03` | réponse reconnue et au moins une citation cible | réponse non reconnue ; citation retournée non cible | récupération/citation | Le moteur classe les chunks, puis extrait exclusivement depuis le premier (`hybrid_rank`, puis `ExtractiveAnswerProvider.answer`, commit `079118f15d52:apps/api/evidencedesk_api/retrieval.py:205-257, 334-352). L'artefact enregistre précisément une citation non correspondante ; il ne s'agit donc pas d'une simple normalisation de texte. |
| `hold2-a06` | réponse reconnue et citée | `abstained`, sans citation | abstention incorrecte | La branche d'abstention précoce de `ExtractiveAnswerProvider.answer` précède l'extraction (`079118f15d52:.../retrieval.py:305-314`). L'artefact ne conserve ni rang ni motif de branche ; l'attribution plus fine entre les deux conditions de cette branche n'est pas prouvable a posteriori. |
| `hold2-amb01` | état ambigu avec preuves attendues | réponse non reconnue ; citation non cible | combinaison : récupération/citation + abstention/ambiguïté incorrecte | Le moteur v2 ne possède qu'un traitement de conflit daté, puis répond depuis le meilleur chunk (`079118f15d52:.../retrieval.py:316-352`). L'artefact confirme que le statut final n'est pas ambigu et que la citation ne correspond pas. |
| `hold2-amb02` | état ambigu avec preuves attendues | réponse non reconnue ; citation non cible | combinaison : récupération/citation + abstention/ambiguïté incorrecte | Même chemin v2 et même constat archivé que `hold2-amb01`. |

Le runner v2 compte une réponse répondable seulement si statut, contenu
normalisé et citation sont tous valides (`079118f15d52:evals/runner.py:200-207`),
et un cas non répondable seulement selon le statut (`:208-214`). Les quatre
écarts ci-dessus ne sont donc pas créés par la métrique ; ils sont présents dans
les sorties du moteur archivées.

## Cas d'extraction v2

| Élément d'artefact | Résultat attendu | Résultat observé | Classement | Cause démontrée |
|---|---|---|---|---|
| cible d'extraction v2, champ `responsible_people` | une valeur avec citation cible | aucune valeur, aucune citation | extraction trop rigide | `extract_supplier_fields` ne remplit ce champ qu'au travers de reconnaissances de formes limitées (`079118f15d52:apps/api/evidencedesk_api/extraction.py:126-139`). L'artefact contient zéro prédiction pour cette cible ; ce n'est pas un échec de la métrique. |

## Cas de questions v3

| Identifiant | Résultat attendu | Résultat observé | Classement | Cause démontrée dans le chemin d'exécution |
|---|---|---|---|---|
| `v3-h-a01` | réponse reconnue et au moins une citation cible | réponse non reconnue ; citation retournée non cible | récupération/citation | Le chemin v3 restreint éventuellement le corpus, classe les chunks puis extrait du premier (`d8b5dab7:evals/runner.py:213-219`; `d8b5dab7:.../retrieval.py:221-294, 417-434`). L'artefact prouve le désaccord de citation et de réponse. |
| `v3-h-a02` | réponse reconnue et citée | `abstained`, sans citation | abstention incorrecte | Retour anticipé d'abstention avant l'extraction (`d8b5dab7:.../retrieval.py:348-357`). Le détail du sous-motif n'est pas journalisé dans l'artefact ; ne pas l'inférer comme défaut du corpus. |
| `v3-h-a05` | réponse reconnue et citée | `abstained`, sans citation | abstention incorrecte | Même branche anticipée et même limite de traçabilité que `v3-h-a02`. |
| `v3-h-a06` | réponse reconnue et citée | `abstained`, sans citation | abstention incorrecte | Même branche anticipée et même limite de traçabilité que `v3-h-a02`. |
| `v3-h-a07` | réponse reconnue et au moins une citation cible | réponse non reconnue ; citation retournée non cible | récupération/citation | Sélection du premier chunk puis réponse extractive, comme pour `v3-h-a01` (`d8b5dab7:.../retrieval.py:417-434`). |
| `v3-h-a09` | réponse reconnue et citée | `abstained`, sans citation | abstention incorrecte | Même branche anticipée et même limite de traçabilité que `v3-h-a02`. |
| `v3-h-u02` | abstention | réponse avec citation non cible | combinaison : abstention incorrecte + récupération/citation | La garde d'absence de preuve n'a pas interrompu le chemin, qui a ensuite répondu depuis le meilleur chunk (`d8b5dab7:.../retrieval.py:348-357, 417-434`). |
| `v3-h-u05` | abstention | réponse avec citation non cible | combinaison : abstention incorrecte + récupération/citation | Même chemin et même constat archivé que `v3-h-u02`. |
| `v3-h-adv02` | abstention | réponse avec citation cible | abstention incorrecte | La garde de sûreté de `ExtractiveAnswerProvider.answer` n'a pas produit le statut requis, puis le chemin normal a répondu (`d8b5dab7:.../retrieval.py:324-357, 417-434`). La citation valide ne rend pas la divulgation acceptable. |

## Cas d'extraction v3

Les 12 erreurs suivantes sont toutes des absences strictes dans l'artefact :
la cible attend une valeur citée ; la sortie contient zéro valeur et zéro
citation. Elles relèvent du moteur d'extraction, pas de la récupération, car
l'extracteur reçoit tous les chunks du document (`d8b5dab7:evals/runner.py:270-275`).

| Cible v3 | Champ | Classement | Cause démontrée |
|---|---|---|---|
| 1 | `organization_name` | extraction trop rigide | Reconnaissance de libellé/forme limitée dans `extract_supplier_fields` et ses aides (`d8b5dab7:apps/api/evidencedesk_api/extraction.py:49-80, 83-162`). |
| 2 | `effective_date` | extraction trop rigide | Même chemin. |
| 3 | `organization_name` | extraction trop rigide | Même chemin. |
| 4 | `effective_date` | extraction trop rigide | Même chemin. |
| 5 | `responsible_people` | extraction trop rigide | Même chemin. |
| 6 | `organization_name` | extraction trop rigide | Même chemin. |
| 7 | `renewal_date` | extraction trop rigide | Même chemin. |
| 8 | `organization_name` | extraction trop rigide | Même chemin. |
| 9 | `effective_date` | extraction trop rigide | Même chemin. |
| 10 | `responsible_people` | extraction trop rigide | Même chemin. |
| 11 | `organization_name` | extraction trop rigide | Même chemin. |
| 12 | `effective_date` | extraction trop rigide | Même chemin. |

## Séparation moteur / runner-métrique

### Faiblesses du moteur constatées

- v2 : 3 sélections avec citation non cible, 3 erreurs d'abstention ou
  d'ambiguïté, et 1 valeur d'extraction absente.
- v3 : 4 sélections avec citation non cible, 7 erreurs d'abstention, et 12
  valeurs d'extraction absentes.
- Aucun de ces constats n'est réductible à une normalisation de vérité attendue
  : chaque réponse incorrecte contient soit un statut incompatible, soit une
  citation non cible, soit les deux ; chaque extraction erronée est absente.

### Limites du runner et de la métrique (non attribuées comme cause des sorties)

- Pour les cas non répondables, le runner juge l'acceptabilité depuis le statut
  seul ; il ne vérifie pas que les preuves attendues d'un cas ambigu soient
  retournées (`079118f15d52:evals/runner.py:208-214` et
  `d8b5dab7:evals/runner.py:237-243`). Une bonne métrique d'abstention ne
  sépare donc pas la qualité des preuves de la seule étiquette de statut.
- Le test de réponse répondable conjonctionne contenu et citation, puis publie
  une seule exactitude par cas ; il ne décompose pas le rang, le choix de phrase
  et le contrôle de citation (`079118f15d52:evals/runner.py:195-207` ;
  `d8b5dab7:evals/runner.py:224-236`). L'artefact permet d'établir les
  catégories ci-dessus, mais pas de départager davantage les sous-causes des
  abstentions sans trace de classement.
- v2 n'inscrit pas ses hashes de manifeste/corpus dans le résultat lui-même ;
  la preuve se trouve dans le verrou et l'attestation. v3 les inscrit dans les
  deux (`d8b5dab7:evals/runner.py:287-300`). C'est une limite d'archivage v2,
  non une modification ou une invalidation des données vérifiées ici.

## Décompte des cas analysés

| Version | Questions erronées uniques | Erreurs extraction uniques | Récupération/citation | Abstention/ambiguïté incorrecte | Extraction trop rigide | Formulation/normalisation seule | Combinaisons |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 | 4 | 1 | 3 | 3 | 1 | 0 | 2 |
| v3 | 9 | 12 | 4 | 7 | 12 | 0 | 2 |

Les colonnes de catégories peuvent dépasser les erreurs uniques, car une même
question peut relever simultanément de l'abstention et de la
récupération/citation. Aucun réglage n'est dérivé de ces holdouts ouverts.

## Suite de protocole v4 à v6 (sans réinterpréter v2/v3)

Les erreurs v2/v3 ci-dessus sont des erreurs historiques détaillées, pas des paramètres de réglage. Les tentatives suivantes sont consignées pour distinguer un échec de protocole d'un résultat moteur : v4 s'est arrêté au préflight parce que l'attestation ne fournissait pas les hashes exigés; aucun raw n'a été créé et le lock est conservé. v5 s'est arrêté dans `evaluate_manifest.build_chunks` avec `KeyError: filename`, avant toute évaluation de cas et sans raw; le lock est conservé. Le validateur contrôle désormais qu'un `filename` non vide est présent dans chaque document avant le runner.

v6 est le premier raw de récupération : son SHA-256 est `033ac62953fe8e447443f69cb185080a17e72989f2f00b706686a6efefd05d03`; le recalcul séparé référencé par cet artefact a le SHA-256 `5d920dc073b5d839be3125188388743974f1434e5de26f229b304df5b46ea6ed` et confirme `FAIL`. Cela ne modifie pas les constats v2/v3 et ne prouve pas un scellement indépendant : les locks restent locaux et non WORM.
