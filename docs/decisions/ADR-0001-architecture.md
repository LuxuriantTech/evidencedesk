# ADR-0001 — Monolithe API, worker séparé et PostgreSQL/pgvector

## Décision

Conserver FastAPI comme API REST, React/TypeScript comme client, PostgreSQL/pgvector comme base de vérité, Redis/ARQ pour les tâches et un stockage local derrière le protocole `DocumentStorage`.

## Raisons

Cette séparation rend visible l’état asynchrone et isole le traitement de la requête HTTP. PostgreSQL réunit métadonnées, audit, recherche textuelle et vecteurs ; le protocole de stockage limite le changement nécessaire pour un adaptateur S3.

## Conséquences

Le compose de développement est reproductible, mais il ne devient pas une architecture de production par simple exposition Internet : TLS, IAM, stockage durable, sauvegardes, rétention et exploitation restent à fournir.
