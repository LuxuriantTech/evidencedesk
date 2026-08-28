# Revue de sécurité locale limitée

## Limite de la revue

Le **Codex Security Deep Scan n'a pas été exécuté par décision utilisateur**. Aucun rapport,
couverture ou verdict de Deep Scan n'existe pour cette publication. Les tests, Gitleaks, Trivy et
audits de dépendances décrits ci-dessous sont des contrôles locaux limités ; ils ne remplacent pas
un Deep Scan ni une revue indépendante.

Une revue Standard locale historique avait une couverture déclarée `partial`. Ses artefacts
canoniques ne sont pas publiés dans ce dépôt ; elle n'est donc pas utilisée comme preuve
indépendamment recalculable de la release. Les deux défauts qu'elle avait aidé à identifier restent
documentés parce que leurs tests de non-régression sont, eux, versionnés.

## Défauts historiques corrigés

| Finding | Sévérité | Confiance | État release candidate |
|---|---|---|---|
| Absence de quota de requêtes sur authentification, question et import | faible | élevée | corrigé et testé |
| Instruction documentaire restituable comme réponse sourcée | faible | élevée | corrigé et revu adversarialement |

Ce tableau ne signifie pas que le dépôt est exempt de vulnérabilités. Il décrit seulement deux
défauts connus et leurs correctifs bornés.

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

## Contrôles de la publication du 28 août 2026

- `pip-audit --strict` : aucune vulnérabilité Python connue ;
- `npm audit --omit=dev --audit-level=high` : zéro vulnérabilité ;
- Gitleaks sur tout l'historique Git et sur l'arborescence locale : zéro secret détecté ;
- Trivy configuration : zéro critique/élevée après passage de l'image web à l'utilisateur
  `nginx` non-root ;
- Trivy web Alpine : zéro critique/élevée ;
- Trivy API Debian brut : 16 occurrences, 13 CVE système uniques critiques/élevées, toutes sans
  version corrigée publiée dans la base utilisée ; scan `--ignore-unfixed` : zéro corrigeable.

Les 13 CVE ne sont pas masquées : leur présence, les paquets concernés et l'absence de chemin
applicatif observé sont analysés dans [`docs/security.md`](security.md). Cette analyse n'annule pas
la dette de base image et impose un nouveau scan avant toute exposition Internet.
