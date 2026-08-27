# Politique de sécurité

EvidenceDesk est un projet de démonstration locale. Il ne revendique pas de SLA ni d’engagement de délai de correction.

> **Research prototype using synthetic data only. Not validated for production, legal, medical,
> financial or compliance decisions.**

Pour signaler une vulnérabilité, utilisez le mécanisme **GitHub Security Advisories** du dépôt concerné une fois celui-ci publié, plutôt qu’une issue publique. Incluez une description minimale, la version/commit concerné, les étapes de reproduction et l’impact observé. Le modèle local [`docs/security-report-template.md`](docs/security-report-template.md) peut être utilisé avant publication. N’incluez jamais de secret, token, document personnel ni jeu de données réel.

Avant publication du dépôt, ne diffusez pas la vulnérabilité ni une preuve d’exploitation : conservez-la localement et demandez au mainteneur un canal privé. Les rapports publics peuvent être confirmés ou corrigés quand une remédiation est disponible, sans garantie de calendrier.

Le périmètre utile comprend authentification/autorisation, import, stockage, suppression, API, worker, dépendances et configuration Docker. Les données synthétiques et les configurations locales de démonstration ne doivent pas être réutilisées en production.

Une injection documentaire révélée par le holdout v7 a été corrigée après l'évaluation et couverte
par des tests adversariaux. Cela ne constitue pas une garantie contre toutes les prompt injections.
Le modèle de menace et les limites du correctif sont documentés dans
[`docs/threat-model.md`](docs/threat-model.md) et [`docs/security.md`](docs/security.md).
