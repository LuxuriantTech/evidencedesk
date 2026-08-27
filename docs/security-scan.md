# Revue Codex Security locale

## Périmètre et intégrité

Le scan standard local `780bfd7e-f2e8-4bfc-ac95-7410d80960b1` a ciblé le commit initial
`7a836458984ce3f79ebb8e64912bb735682a7b6b`. Le rapport scellé a le SHA-256
`2ff93bcdcc60ae91f3844cb486b82f04f61a40547872b3c95536b9926766ba8e` et le fichier canonique
des findings `b1bb2078b7bec21c4693f6f744fa7196695fd9d9b95ba233104450e01fdfcb55`.

Le scan est terminé mais sa couverture canonique est `partial` : il s'agit d'une revue locale
standard, pas d'une certification exhaustive. Les anciens holdouts ont été lus comme preuves sans
être exécutés. Le worktree a changé pendant le scan parce que les remédiations autorisées étaient
développées en parallèle ; les deux findings ci-dessous décrivent donc le commit initial.

## Findings du commit initial

| Finding | Sévérité | Confiance | État release candidate |
|---|---|---|---|
| Absence de quota de requêtes sur authentification, question et import | faible | élevée | corrigé et testé |
| Instruction documentaire restituable comme réponse sourcée | faible | élevée | corrigé et revu adversarialement |

Aucun finding critique, élevé ou moyen n'a été enregistré par ce scan. Cela signifie uniquement
qu'aucun n'a été trouvé dans ce périmètre et avec cette couverture ; ce n'est pas une preuve
d'absence de vulnérabilité.

## Vérification des corrections

La limitation de débit applique une fenêtre glissante bornée, renvoie `429` avec `Retry-After` et
ignore `X-Forwarded-For`. L'overlay public utilise des quotas plus stricts et désactive les comptes
administrateur/lecteur. Sa limite résiduelle est documentée : le compteur est mono-processus et le
proxy fourni peut faire partager un bucket entre visiteurs.

La frontière de confiance sépare règles système, question utilisateur, contenu documentaire non
fiable et preuve. Une revue adversariale indépendante a reproduit puis fait corriger les variantes
multi-lignes, ponctuées, Unicode, les anciennes extractions sans preuve survivante et une instruction
adressée à l'assistant mais présentée comme obligation contractuelle. Les contrepreuves
opérationnelles normales restent admises. Le verdict final de cette revue est : remédié pour les
canaux vérifiés, avec détection déterministe bornée et sans garantie universelle.

Le résultat v7 n'a pas été rejoué ni rescorré après ces corrections.

## Scans complémentaires de la release

- `pip-audit --strict` : aucune vulnérabilité Python connue ;
- `npm audit --omit=dev --audit-level=high` : zéro vulnérabilité ;
- Gitleaks sur l'arborescence locale : zéro secret détecté ;
- Trivy web Alpine : zéro critique/élevée ;
- Trivy API Debian brut : 16 occurrences, 13 CVE système uniques critiques/élevées, toutes sans
  version corrigée publiée dans la base utilisée ; scan `--ignore-unfixed` : zéro corrigeable.

Les 13 CVE ne sont pas masquées : leur présence, les paquets concernés et l'absence de chemin
applicatif observé sont analysés dans [`docs/security.md`](security.md). Cette analyse n'annule pas
la dette de base image et impose un nouveau scan avant toute exposition Internet.
