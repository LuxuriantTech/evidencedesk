# Contribuer à EvidenceDesk

## Pré-requis

Python 3.12 avec `uv`, Node.js 24 et Docker Compose sont utilisés par les contrôles locaux. Ne placez aucun document personnel, secret, export de production ou fichier `.env` dans le dépôt.

## Boucle locale

```bash
uv sync --all-groups
uv run ruff check apps/api apps/worker evals scripts
uv run mypy
uv run pytest
cd apps/web && npm ci && npm run lint && npm run typecheck && npm run test && npm run build
```

Pour le parcours complet : `docker compose up --build --wait` puis consultez `http://127.0.0.1:8080`. Arrêtez avec `docker compose down`.

## Règles de contribution

- Travaillez avec des données exclusivement synthétiques.
- Ajoutez un test qui échoue avant tout correctif comportemental, puis le test ciblé et la suite pertinente.
- Préservez les citations document/page/extrait et l’abstention ; ne remplacez pas une absence de preuve par une réponse plausible.
- Pour un changement d’algorithme, mettez à jour les évaluations de développement. Ne rejouez jamais un holdout déjà verrouillé.
- Les changements de schéma nécessitent une migration Alembic et un test d’intégration.
- Exécutez les contrôles listés ci-dessus avant une pull request. La CI exécute aussi les audits de dépendances, Gitleaks et la stack Docker.

## Signalement de sécurité

N’ouvrez pas un ticket public pour une vulnérabilité : consultez [SECURITY.md](SECURITY.md).
