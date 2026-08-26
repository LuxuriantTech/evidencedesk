# Architecture

EvidenceDesk est une application locale de traitement de dossiers opérationnels synthétiques. Le parcours livré est : connexion, import, traitement asynchrone, recherche avec sources, extraction structurée, audit et évaluation.

```mermaid
flowchart LR
  W[React + TypeScript\nNginx :8080] -->|/api/v1| A[FastAPI]
  A -->|JWT + RBAC| P[(PostgreSQL + pgvector)]
  A -->|tâche ARQ| R[(Redis)]
  R --> K[Worker ARQ]
  K --> P
  A --> S[Stockage local\n/data/documents]
  K --> S
  A --> M[/health /ready /metrics]
```

## Composants et responsabilités

- `apps/web` : React et TypeScript. L’interface consomme l’API REST derrière Nginx et n’accorde aucune permission par elle-même.
- `apps/api/evidencedesk_api/main.py` : FastAPI, authentification, contrôle serveur des rôles, import, recherche, audit, évaluation et endpoints d’observabilité.
- `apps/worker/evidencedesk_worker/jobs.py` : worker ARQ. Il lit une tâche Redis, extrait le texte, masque les motifs PII synthétiques, produit les chunks et l’extraction, puis met à jour l’état.
- PostgreSQL : métadonnées, audit, tâches, extractions et chunks. Les vecteurs `Vector(384)` ont un index HNSW ; le texte a un `tsvector` et un index GIN.
- Redis : file ARQ, pas une source durable de vérité métier.
- `LocalDocumentStorage` : fichiers sous une racine résolue et clés construites à partir d’un UUID. L’interface `DocumentStorage` (`put`, `get`, `delete`) permet d’ajouter un adaptateur S3 sans modifier les routes ni le worker.

## Données et traitement

L’import accepte uniquement PDF textuel, TXT et Markdown. Un middleware ASGI borne le corps HTTP complet à 11 MiB avant le parseur multipart, même sans `Content-Length`; le handler borne ensuite le fichier à 10 MiB et vérifie extension, type annoncé, signature `%PDF-`, UTF-8, contenu vide et octet NUL. En mode démo publique, le SHA-256 doit aussi appartenir à la liste synthétique versionnée, pour l’upload comme pour le seed ; une attestation client seule ne suffit pas. Le nom est réduit à son basename et la clé de stockage est `UUID/document.extension` : un chemin utilisateur ne pilote jamais le stockage.

Une contrainte d’unicité partielle `(dossier_id, content_sha256)` empêche un second document actif identique. Chaque document possède une unique tâche, un identifiant de corrélation, des états `queued`, `processing`, `completed`, `failed` ou `deleted`. Le worker est idempotent : un document déjà terminé ne recrée pas ses chunks. La suppression et tous les chemins worker verrouillent tâche puis document ; le marqueur `deleted` est revérifié avant persistance et avant mise à jour d’échec. Les erreurs de stockage et de traitement ont une reprise ARQ contrôlée ; après les tentatives prévues, la tâche est marquée en échec avec un code court.

Après la limite d’import, trois budgets bornent l’expansion : 200 pages, 1 000 000 de caractères extraits et 5 000 chunks par défaut. Un dépassement est terminal avec un code stable, sans reprise inutile. Pour PDF, `pypdf` tourne dans un sous-processus `spawn` avec délai mural ; sous POSIX, `RLIMIT_AS` et `RLIMIT_CPU` ajoutent des plafonds d’espace d’adressage et de CPU. La limite mémoire dure n’est pas disponible sous Windows natif.

Les chunks conservent page, section, ordinal et texte masqué. La recherche combine candidats vectoriels et lexicaux, puis un classement hybride. Une réponse contient les passages retenus, le document et la page. En cas de preuve insuffisante ou contradictoire, le fournisseur extractif renvoie une abstention explicite.

## Modes de réponse

Le mode livré est `extractive-local`. Son vecteur est un feature hashing déterministe (tokens et trigrammes), de dimension 384, complété par la recherche lexicale PostgreSQL. Il est reproductible, gratuit et ne nécessite pas de clé, mais **ce n’est ni un embedding sémantique pré-entraîné ni un LLM complet**. Ses résultats sont surtout sensibles au recouvrement lexical et aux formulations du corpus.

`ProviderBundle` regroupe les protocoles `EmbeddingProvider` et `AnswerProvider`. Le registre refuse
tout mode inconnu ; un fournisseur local ou compatible OpenAI peut être ajouté à ces deux frontières
avec un mode et un coût explicites. Aucun appel externe ni coût n’est effectué par défaut.

## Accès et observabilité

Les mots de passe sont vérifiés côté API et stockés sous forme de hash Argon2. Les jetons JWT sont courts (30 minutes par défaut). Les rôles sont appliqués au niveau de chaque route : administrateur (toutes actions), analyste (import, lecture, questions, extraction) et lecteur (lecture, questions, extraction). L’audit enregistre acteur, action, résultat, document éventuel, date et identifiant de corrélation.

Le middleware renvoie `X-Correlation-ID` et produit des logs JSON sans corps de document, mot de passe ou token. Prometheus est exposé sous `/metrics`; `/health` vérifie le processus et `/ready` vérifie PostgreSQL et Redis. `/api/v1/status` est réservé à l’administrateur.

## Déploiement local

`compose.yaml` compose PostgreSQL/pgvector, Redis, migration, seed synthétique, API, worker et web. Les ports publiés sont liés à l’interface loopback ; le volume `documents_data` est partagé seulement entre API et worker. Les images externes et les actions CI sont épinglées à des révisions immuables. Chaque service basé sur l’image locale API déclare le même build et `pull_policy: build`, afin qu’un lancement ciblé normal reconstruise avant exécution. `scripts/check_supply_chain_refs.py` contrôle ces invariants. Nginx limite les requêtes à 11 MiB, ajoute `nosniff`, `DENY` contre l’iframe, `no-referrer` et une CSP restrictive. Ces contrôles ne remplacent pas TLS, un reverse proxy de production, la rotation des secrets, une politique IAM S3 ou une vérification de provenance des images.
