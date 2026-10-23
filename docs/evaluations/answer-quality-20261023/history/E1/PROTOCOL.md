# Protocole de comparaison indépendant du développement

Version du jeu : `ind23-comparison-cases-v1`. Corpus et attentes rédigés avant de recevoir les sorties d'évaluation du moteur. Le périmètre est synthétique et local. Aucun service externe ni API payante n'est requis.

## Gel et usage

Les fichiers sous `datasets/` sont le gel initial. Le jeu contient 20 questions : 13 répondables, 5 sans réponse suffisante et 2 ambiguës. Les réponses attendues, passages exacts, catégories et groupes de paraphrases sont définis avant toute exécution. Ne pas modifier questions, catégories, citations, seuils, règles de succès ou réponses tolérées après lecture d'une sortie. En cas d'erreur du jeu, arrêter et publier une nouvelle version au lieu de modifier celle-ci silencieusement.

Comparer baseline et candidat avec exactement les mêmes corpus et questions, le même mode, le même runner, les mêmes dépendances, le même environnement et les mêmes paramètres. Archiver configuration, commit/code fingerprint, version modèle, sortie brute, heure et sommes SHA-256 avant d'adjudicer les réponses. La baseline et le candidat peuvent être exécutés séparément, mais aucun seuil ou attente ne peut être retouché entre les deux.

Cette séparation entre auteur des questions et agent qui modifie le runtime réduit les risques de fuite pendant le développement. Elle ne constitue pas une certification par un véritable tiers aveugle : l'auteur connaît les documents et les réponses, le dossier est synthétique, et les agents travaillent dans un même projet. Ne pas appeler ces cas un holdout institutionnel ou une preuve de généralisation à des documents réels.

## Règles de jugement préalables

- Répondable : la réponse doit contenir le fait exact et une citation vers le document, la page et le passage prévus. Pour les dates et montants, aucune valeur proche, conversion ou alternative non sourcée n'est acceptée.
- Sans réponse : le statut doit être une abstention explicite ; toute date, valeur ou fait inventé constitue un échec. Une phrase indiquant que le dossier ne fournit pas la réponse est acceptable.
- Ambigu : le statut doit signaler l'ambiguïté. Pour la collision contractuelle, la sortie doit présenter 24 h sous le MSA, 12 h pour une violation de données personnelles sous l'addendum, et indiquer que le corpus ne précise pas la priorité entre ces clauses. Pour la date d'effet générique, nommer les deux entités et leurs dates, puis demander laquelle intéresse l'utilisateur. Choisir un seul fait sans contexte échoue.
- Paraphrase : au sein d'un même `paraphrase_group`, garder la même valeur et la même preuve malgré la formulation et la langue.
- Négation : vérifier explicitement la polarité. Dire qu'un sous-traitant peut agir sans accord échoue ; omettre la négation dans la réponse au rapport d'incident échoue.
- Ne pas utiliser la précision du retrieval, le nombre de citations ou l'absence d'exception comme substitut à l'exactitude de la réponse.

Rapporter séparément l'exactitude répondable, la précision/rappel de citation, l'abstention, le traitement des ambiguïtés, les performances par catégorie (paraphrase, négation, valeurs proches, collisions) et les erreurs techniques. Les dénominateurs doivent rester visibles. L'extraction structurée n'est pas évaluée par ce jeu (`extraction_targets` vide) ; ne pas interpréter une métrique d'extraction vide comme un résultat mesuré. N'établir aucun seuil de réussite après avoir observé les résultats ; en l'absence d'un seuil préenregistré, publier les mesures comme descriptives.

## Origine

Les questions reprennent des types de faits présents dans les contrats et rapports synthétiques du dépôt (dates d'effet/renouvellement, frais avec périodes, obligations, notice d'incident, données absentes), puis les appliquent à des documents fictifs rédigés spécialement pour cette version. Aucun cas, valeur ou document de v7 n'est copié. Le corpus indépendant sert uniquement de source des faits gold de ce jeu.
