# Évaluation d'EvidenceDesk

Les évaluations locales mesurent recherche, réponse sourcée, abstention et extraction sur corpus
synthétiques versionnés. Elles ne mesurent ni des données réelles, ni une charge API/SQL à grande
échelle.

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

## Développement v3 — réponse et extraction

Le développement v3 est nouveau et n'utilise pas les gold des holdouts v2 à v6. Il contient huit
documents synthétiques, 48 cas et deux partitions de 24 cas séparées par familles de modèle et de
formulation : 16 répondables et 8 non répondables par partition. Les 48 cas comprennent 32
répondables, 8 sans réponse, 4 ambigus et 4 adversariaux. Les règles de génération et les groupes
sont dans `datasets/development_v3/generation_manifest.json` ; le corpus, les cas et ce manifest ont
respectivement pour SHA-256 `d61e15…fe08`, `3caa19…4139` et `b67bcc…f595`.

Le plan `answer-v3-experiment-plan.json` a borné l'expérience à trois stratégies, deux
configurations chacune et trois répétitions déterministes. Qwen a été rejeté après calibration pour
erreurs de schéma/techniques ; sa partition de sélection n'a donc pas été ouverte par le harness.

| Configuration | Répondable sélection | Abstention sélection | Citation P/R | Extraction F1 | Erreur/schéma | p95 sélection |
|---|---:|---:|---:|---:|---:|---:|
| déterministe A | 0,9375 | 0,75 | 1 / 1 | 0,931818 | 0 / 0 | 677,929 ms |
| déterministe B | 0,9375 | 0,75 | 1 / 1 | 0,931818 | 0 / 0 | 728,256 ms |
| NLI A | 0 | 0,625 | 1 / 1 | 0,931818 | 0 / 0 | 830,019 ms |
| NLI B | 0 | 0,625 | 1 / 1 | 0,931818 | 0 / 0 | 833,753 ms |
| Qwen A | non ouvert | non ouvert | calibration 0 / 0 | calibration 1 | calibration 0,791667 / 0,791667 | 10 859,263 ms calibration |
| Qwen B | non ouvert | non ouvert | calibration 1 / 0,5 | calibration 1 | calibration 0,75 / 0,75 | 14 586,243 ms calibration |

La règle maximin préenregistrée retient `deterministic-v3-a`. Une reproduction finale avec le
runtime et le schéma v4 donne les mêmes décisions sur trois exécutions. Sur la sélection : Recall@5
`1`, MRR@5 `0,96875`, répondable `0,9375`, abstention `0,75`, citation P/R `1/1`, extraction
P/R/F1 `0,953488/0,911111/0,931818`, erreur et schéma `0`. Sur la reproduction finale,
la médiane agrégée est `836,865 ms`, le p95 `1 026,154 ms` et le pic RSS
`904 192 000` octets.

Le verdict de développement reste **FAIL**, car `0,75 < 0,85` pour l'abstention. Aucune règle n'a
été modifiée après ouverture de la partition de sélection. Le résumé source est
`artifacts/evaluations/development_v3/frozen-runtime-v4-locked-final/summary.json` (SHA-256
`555975…9caf`) ; le détail de décision est dans l'ADR-0006.

Deux suites antérieures sont conservées comme incidents de protocole et ne servent pas au gel :
`strategy_v3-before-final-protocol-binding/` précède les validations fail-closed finales ;
`strategy_v3-before-relative-path-fix/` a été invalidée lorsqu'un chemin relatif a révélé un défaut
du finalizer. L'essai final incomplet correspondant est conservé dans
`frozen-runtime-v4-locked-final-relative-path-failure/`. Après ajout du test de régression, les six
configurations ont été rejouées sous l'empreinte courante avant la reproduction finale.

## Correctifs du protocole avant v7

Le préflight générique valide maintenant schéma, attestation, nom de fichier, dataset, gold et
configuration, puis construit les fournisseurs v3 avant de pouvoir créer le lock. Un échec avant
inférence ne consomme donc plus l'exécution aveugle. Les erreurs de v4 et v5 restent immuables comme
incidents historiques.

### Garde supplémentaire ajouté après v7

Une revue indépendante postérieure à v7 a trouvé un autre chemin ordinaire : l'API Python publique
`evaluate_manifest` acceptait encore `split="holdout"` sans passer par le préflight attesté, même si
la CLI le refusait déjà. Ce chemin n'est pas celui enregistré dans le raw v7, mais l'absence de garde
était un défaut réel. Il est maintenant refusé avant toute lecture de dataset ; le runner attesté
appelle une continuation interne dédiée après préflight et création du lock. Les runners historiques
v4 à v6 refusent désormais toute invocation avant lecture ou écriture. Ces gardes ferment les points
d'entrée supportés et préviennent un contournement accidentel, pas un opérateur local capable
d'importer des fonctions privées ou de modifier le code.

Ce correctif post-v7 change l'empreinte du moteur courant sans réécrire le moteur figé, la
configuration, le lock ou le raw v7. La validation historique recalcule désormais l'empreinte depuis
les objets Git du commit moteur `84971d8c01ac0e0eac840205527acad0a92561e7`; le contrôle pré-exécution
continue, lui, d'exiger que les sources courantes correspondent exactement à une nouvelle
configuration figée.

Le runner courant émet le schéma générique `evidencedesk-attested-holdout-raw-v1` et le
recalculateur `evidencedesk-recalculated-evaluation-v1`. Le recalculateur accepte aussi une liste
fermée de schémas historiques immuables, rejette un schéma inconnu et recalcule Recall@5/MRR@5 à
partir des documents/pages récupérés et des locations gold, sans faire confiance à un booléen
pré-calculé. Il vérifie aussi la correspondance des latences par cas, la validité de l'indexation et,
lorsque ces champs existent au niveau racine, le temps total et le pic RSS.

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

## Résultat holdout v7 — seul raw enregistré

Le protocole v7 a été gelé au commit `e9af96e3ea2a92d567bd9e9d771fae33b7e5b684`, puis le dataset
indépendant a été committé dans `152e7ee2248cc00f8692c907a0b0f7830727e2ba`. Après préflight réussi,
le lock `03b455f0485d04afce46cf5798ce537786169a34d9b51787a372e121e01ac909` a été créé le
2026-08-27T18:49:52.648828+00:00. Il lie le moteur `84971d8c01ac0e0eac840205527acad0a92561e7` au
dataset versionné `blind-holdout-v7-2026.08.27`. Le graphe Git local contient un seul lock et un
seul raw v7 ; aucun v8 ne sera créé dans ce cycle. Ce constat ne peut pas exclure un éventuel run
local supprimé avant commit, car le verrou n'est ni WORM ni attesté par un tiers.

Le raw est `FAIL` : précision/rappel de citation `0,8 / 0,48` (12 citations correctes sur 15
retournées pour les 25 cas répondables, 12 preuves gold retrouvées sur 25), exactitude répondable
`0,36` (9/25), abstention `0,8` (12/15), extraction P/R/F1
`0,659091 / 0,349398 / 0,456693` (29 vrais positifs, 44 prédictions, 83 valeurs gold), Recall@5 `1`
(25/25), MRR@5 `0,94`, erreurs et erreurs de schéma `0`, coût externe `0 USD`. Les objectifs gelés
de 90 % pour citations/cas répondables/F1 et 85 % pour l'abstention ne sont pas atteints. Vingt
citations sont émises tous types de cas confondus ; les compteurs officiels de précision/rappel de
citation ne portent que sur les 25 cas répondables. Le taux d'erreur nul ne mesure que les
exceptions techniques : il coexiste avec 19/40 décisions sémantiquement incorrectes.

Temps runner local : indexation `3778,661 ms`, médiane `756,641 ms`, p95 `1150,454 ms`, total
`36288,053 ms`; pic RSS `700862464` octets. Ce ne sont pas des SLO de service. Le runtime enregistré
est CPU (`12th Gen Intel(R) Core(TM) i5-12600KF`, 16 CPU logiques, `16768458752` octets RAM), Python
`3.12.13`, WSL2, ONNX Runtime `CPUExecutionProvider`.

Le recalcul séparé à partir du raw, sans inférence ni réouverture du corpus, reproduit ces agrégats,
le Recall@5/MRR@5 et la cohérence interne des temps/RSS. Il réutilise les décisions de valeur et de
citation enregistrées dans le raw : ce n'est pas une nouvelle évaluation sémantique. Les SHA-256
sont : raw
`55a79af0d38ea1d7742763406e29dc96416f7c3cb158aa9a90bbbbd16650c1c1`, recalcul
`fd9e2864cd6b2aa4701cc1e2b979d49cd02e4fea76fbe8727271a45c9585fee3`, corpus
`32f0d0bda0abdc94cf2477dd640f82a7b37b7b2062bdc512d5480d5ac267d574`, cas
`2ab6c710a2eed039ae8749f151eb5894f875dd1411b69fcec9b106571580b1a1` et attestation
`b60b2a74e69ca52a67117d610d15a5d7b95c5d131b478a5fa74fbe00d8ed9021`.

Recalcul vérifiable sans appeler le moteur :

```bash
PYTHONPATH=apps/api:apps/worker:. uv run python -m evals.recalculate \
  --input artifacts/evaluations/holdout_v7/raw.json \
  --output /tmp/holdout-v7-recalculated.json
sha256sum artifacts/evaluations/holdout_v7/raw.json \
  artifacts/evaluations/holdout_v7/recalculated.json \
  artifacts/evaluations/holdout_v7/grounded-local-v3.0-frozen-v7.lock.json
```

Le diagnostic est post-récupération : 11 questions répondables ont été refusées alors que leur
preuve est dans le top-5 ; `v7-a04` choisit une mauvaise personne, `v7-a12` transforme une date de
renouvellement en ambiguïté, `v7-a21` et `v7-a23` ne sont que partiellement soutenues. Deux des trois
questions ambiguës reçoivent encore une réponse certaine. Le cas adversarial `v7-x02` restitue une
instruction injectée : c'est un échec de résistance aux injections. Le cas `v7-a16` a une citation
correcte et contient le montant attendu, mais le normaliseur retient d'abord l'horodatage `06:23` ;
cette limite du scoreur est documentée après ouverture et ne corrige pas le verdict (même une unité
supplémentaire resterait très sous le seuil).

### Correctifs de release postérieurs à v7 — sans rescoring

La release candidate corrige deux défauts logiciels révélés par v7 : le normaliseur de montants ne
prend plus un horodatage sans devise pour un montant, et une frontière de confiance commune refuse
les instructions documentaires dans réponses, citations, évaluations candidates et extractions.
Les régressions couvrent français/anglais, divulgation, exfiltration, faux administrateur, faux
résultat, contradiction, segmentation multi-ligne, ponctuation et homoglyphes courants.

Ces changements sont postérieurs au lock/raw. Ils ne modifient ni le corpus, ni le raw, ni le
recalcul, et v7 n'a pas été réexécuté. Ils ne constituent donc aucune amélioration mesurée de v7 et
ne changent pas le verdict `FAIL`.

L'extraction confirme le défaut de généralisation : type et organisation `0/10`, date d'effet `1/10`,
montants `7/11`, obligations `6/10`, renouvellement `5/9`, responsables `4/11`, risques `6/12`
(numérateur = vrais positifs). Les formulations de date, les rôles/personnes, les valeurs courtes de
risque et les contradictions restent mal couverts. Voir l'analyse détaillée dans
[`docs/holdout-v7-analysis.md`](holdout-v7-analysis.md).

## Résultat holdout v6 — historique immuable

Le raw v6 est `FAIL` : précision citation `0,4` (2/5), exactitude citation par cas `0,08` (2/25),
abstention `0,8` (12/15), extraction P/R/F1 `1 / 0,291667 / 0,451613`, Recall@5 `1`, MRR@5 `0,9`,
erreurs `0`, coût externe `0 USD`. Il s'abstient sur 20 des 25 cas répondables et traite strictement
0 des 3 cas ambigus comme ambigus. L'extraction compte 35 vrais positifs sur 120 valeurs gold et
35 prédictions.

Temps runner local : indexation `2035,357 ms`, médiane `38,802 ms`, p95 `55,001 ms`, total
`3651,221 ms`; pic RSS `698339328` octets. Le diagnostic historique localise déjà l'échec après la
récupération : preuve gold top-5 pour 25/25, mais décision de répondre, citation et patrons
d'extraction insuffisants. Les vrais positifs par champ sont type `0/10`, date d'effet `5/10`,
montants `20/20`, obligations `0/20`, organisation `0/10`, renouvellement `10/10`, responsables
`0/20` et risques `0/20`. Ce holdout ouvert ne doit servir à aucun réglage.

Recalcul historique, sans moteur :

```bash
PYTHONPATH=apps/api:apps/worker:. uv run python -m evals.recalculate \
  --input artifacts/evaluations/holdout_v6/holdout-v6-raw.json \
  --output /tmp/holdout-v6-recalculated.json
sha256sum artifacts/evaluations/holdout_v6/holdout-v6-raw.json \
  artifacts/evaluations/holdout_v6/holdout-v6-recalculated.json
```

Le raw v6 conserve volontairement le label historique `holdout-v4-raw-result-v2`; le nouveau runner
n'émet plus cette étiquette. Cette correction ne réécrit aucun artefact ouvert.

## Modèle et limites

Le modèle est `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` révision `faf4aa4225822f3bc6376869cb1164e8e3feedd0`, Apache-2.0, environ 118 M paramètres, 384 dimensions, maximum 512 tokens, pooling mean, CPU avec `fastembed==0.8.0` et `onnxruntime==1.29.0`. Téléchargement mesuré déclaré dans le manifest : `266906689` octets, dont `235052644` pour ONNX; pic du run historique de développement v2 : `725581824` octets (~725,6 MB). La reproduction finale v3 a atteint `904192000` octets (~904,2 MB). Les sept fichiers listés totalisent `266903238` octets, soit `3451` de moins que la mesure globale ; le manifest n'attribue pas cet écart. Le pooling mean est une convention fastembed, pas une validation indépendante d'optimalité. L'intégrité locale repose sur manifest et hashes, pas sur une signature de provenance.
