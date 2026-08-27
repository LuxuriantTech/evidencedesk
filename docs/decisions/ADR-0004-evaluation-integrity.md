# ADR-0004 — Intégrité des évaluations et verrouillage local des holdouts

## Décision

Le protocole exige une seule exécution par holdout sous un lock global local. Le résultat conserve
hashes, fingerprint moteur, paramètres, graine et mode. Le lock est indépendant du répertoire de
sortie et bloque une seconde invocation via les points d'entrée supportés tant que lock/raw sont
présents.

## Historique

- v2 et v3 ont produit `FAIL`; l'audit détaillé est dans `docs/holdout_v2_v3_error_audit.md`.
- v4 a échoué au préflight : l'attestation ne contenait pas les SHA-256 requis. Aucun raw n'a été créé et le lock est conservé.
- v5 a échoué avant les cas par `KeyError: filename` dans `evaluate_manifest.build_chunks`. Aucun raw n'a été créé et le lock est conservé. Le validateur de dataset impose désormais un `filename` non vide avant l'exécution.
- v6 a produit une seule sortie raw (`033ac62953fe8e447443f69cb185080a17e72989f2f00b706686a6efefd05d03`) puis un recalcul séparé (`5d920dc073b5d839be3125188388743974f1434e5de26f229b304df5b46ea6ed`), tous deux `FAIL`.
- v7 possède un lock, un raw (`55a79af0d38ea1d7742763406e29dc96416f7c3cb158aa9a90bbbbd16650c1c1`)
  et un recalcul séparé (`fd9e2864cd6b2aa4701cc1e2b979d49cd02e4fea76fbe8727271a45c9585fee3`)
  dans le graphe Git local ; le verdict est `FAIL`.

## Limite de preuve

Un lock local n'est ni signé, ni WORM, ni horodaté par un tiers. Les artefacts établissent une
cohérence locale, pas un scellement indépendant, l'absence d'un run supprimé, ni l'absence de
consultation/tuning préalable.

Une revue postérieure à v7 a aussi trouvé que l'API Python publique du runner pouvait appeler le
split holdout sans le préflight attesté. Le chemin v7 enregistré a bien utilisé le runner attesté,
mais le garde manquait : le correctif post-v7 refuse maintenant cet appel public et désactive les
runners historiques avant lecture ou écriture. La continuation interne reste une convention de code,
pas une frontière contre un processus Python local hostile. Pour préserver la provenance, l'ancien
fingerprint, le manifeste modèle et les dépendances sont recalculés depuis les objets Git du commit
moteur, tandis qu'un futur préflight exige toujours l'égalité avec les sources courantes.

## Interdictions

Il est interdit de retuner contre un holdout ouvert, de remplacer son artefact, de modifier ses cas ou de contourner le lock par un autre chemin. Une amélioration exige un nouveau corpus indépendant et une nouvelle exécution unique.
