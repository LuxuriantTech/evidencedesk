# Préparation portfolio — EvidenceDesk

**Statut : publication du code source autorisée.** Le dépôt dédié est
[`LuxuriantTech/evidencedesk`](https://github.com/LuxuriantTech/evidencedesk). L'application
n'est pas déployée sur Internet ; CV, LinkedIn, Indeed et profils restent hors périmètre.

## Carte projet courte — anglais

### EvidenceDesk

Local evidence-grounded document-review prototype for synthetic supplier operations. It combines a
React/TypeScript interface, FastAPI, PostgreSQL/pgvector and Redis/ARQ, with asynchronous ingestion,
page-level evidence, explicit abstention, PII redaction and correlated audit logs. The active local
retrieval is hybrid and uses a CPU ONNX embedding with no external API key. Its 40-case v7 holdout
is an honest FAIL, so this project demonstrates engineering and evaluation discipline—not validated
general RAG quality.

- Repository: [LuxuriantTech/evidencedesk](https://github.com/LuxuriantTech/evidencedesk)
- Demo: not deployed; local Docker instructions are in the repository
- Architecture: [docs/architecture.md](https://github.com/LuxuriantTech/evidencedesk/blob/main/docs/architecture.md)
- Evaluation: [docs/evaluation.md](https://github.com/LuxuriantTech/evidencedesk/blob/main/docs/evaluation.md)

## Carte projet courte — français

### EvidenceDesk

Prototype local de revue documentaire sourcée pour des dossiers fournisseurs synthétiques.
L'application associe React/TypeScript, FastAPI, PostgreSQL/pgvector et Redis/ARQ ; elle couvre
l'ingestion asynchrone, les preuves par page, l'abstention explicite, le masquage PII et l'audit
corrélé. La recherche active est hybride avec embedding ONNX sur CPU, sans clé API externe. Le
holdout v7 de 40 cas est un FAIL assumé : le projet montre une réalisation technique et une
discipline d'évaluation, pas une qualité RAG générale validée.

- Dépôt : [LuxuriantTech/evidencedesk](https://github.com/LuxuriantTech/evidencedesk)
- Démonstration : non déployée ; lancement Docker local documenté dans le dépôt
- Architecture : [docs/architecture.md](https://github.com/LuxuriantTech/evidencedesk/blob/main/docs/architecture.md)
- Évaluation : [docs/evaluation.md](https://github.com/LuxuriantTech/evidencedesk/blob/main/docs/evaluation.md)

## Limites de publication

Le portfolio ne doit afficher aucun badge PASS global ni suggérer que les objectifs holdout ont été
atteints. Un déploiement applicatif, une modification de CV ou de profil et toute autre publication
restent des actions distinctes.

## Références dans le dépôt

- architecture : `docs/architecture.md` ;
- résultats : `docs/evaluation.md` et `artifacts/evaluations/holdout_v7/` ;
- démonstration : `docs/demo-script.md` ;
- captures : `docs/screenshots/`.
