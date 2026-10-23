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

Lorsqu’une question demande de relier plusieurs documents, le runtime ne joint pas les dossiers sur
la seule coïncidence de noms. Il utilise une référence explicite typée, telle que `(shipment,
SH-2841)`, et exige la même valeur de référence typée dans les documents concernés. Si les sources
présentent plusieurs valeurs incompatibles pour le type de référence demandé, la jointure est
refusée plutôt que devinée. Une référence de contrat commune ne suffit donc pas à prouver que deux
documents concernent le même envoi.

Après la limite d’import, trois budgets bornent l’expansion : 200 pages, 1 000 000 de caractères extraits et 5 000 chunks par défaut. Un dépassement est terminal avec un code stable, sans reprise inutile. Pour PDF, `pypdf` tourne dans un sous-processus `spawn` avec délai mural ; sous POSIX, `RLIMIT_AS` et `RLIMIT_CPU` ajoutent des plafonds d’espace d’adressage et de CPU. La limite mémoire dure n’est pas disponible sous Windows natif.

Les chunks conservent page, section, ordinal et texte masqué. La recherche combine candidats
vectoriels et lexicaux, puis un classement hybride. Une réponse contient les passages retenus, le
document, la page et l'extrait justificatif. Le moteur v3 classe séparément preuve supportée,
support partiel, contradiction et absence de preuve. Sa validation finale refuse une réponse dont
le chunk, le document, la page ou l'extrait ne peuvent pas être reliés aux candidats récupérés.

`trust_boundaries.py` matérialise les couches règles système, question utilisateur, document non
fiable et preuve admise. Le runtime conserve l'instruction documentaire dans la couche de données
non fiable. Avant qu'un fragment traverse une réponse, citation, évaluation candidate ou extraction
persistée, le contexte adjacent est restauré et le garde déterministe est rejoué. Ce contrôle
post-v7 bloque les formes adversariales sur les canaux testés, sans prétendre résoudre
universellement la prompt injection.

### Réponse fondée sur les sources

`DeterministicGroundedAnswerProvider` suit le même pipeline pour les réponses du mode local :

1. Il classe la forme de la question (par exemple date, montant, durée, statut ou oui/non). Les
   demandes qui exigent plusieurs faits distincts sont laissées sans réponse lorsqu'ils ne peuvent
   pas être réunis de manière sûre.
2. Il détermine la portée documentaire à partir des termes de source et du contexte. Un document
   simplement nommé comme cible d'une question ne devient pas automatiquement la source exclusive.
3. Il repère dans les passages les valeurs et relations pertinentes, en tenant compte de leur clause
   locale et des conditions indiquées, comme le type d'incident auquel un délai s'applique.
4. Il traite une négation en lien avec la relation recherchée. Une coordination de sujets qui partage
   un prédicat conserve cette portée (« couverture et maintenance ne sont pas incluses »); une
   coordination de deux propositions ne les confond pas si chacune a son propre prédicat. Deux
   valeurs ne sont contradictoires que si elles répondent à la même relation dans la portée
   applicable.
5. Il retourne l'extrait vérifiable et ses références. Une preuve insuffisante ou des sources
   réellement incompatibles entraînent une abstention ou un résultat ambigu.

Cette logique déterministe utilise des règles linguistiques limitées et des équivalences de termes;
ce n'est pas une analyse complète du langage. Elle peut manquer des formulations inédites,
des références implicites ou des relations exprimées de façon complexe. Le score d'appui interne
sert au seuil de décision, mais n'est pas calibré : il ne doit pas être interprété ni affiché comme
une probabilité de justesse. L'interface présente maintenant l'appui des sources sur 100. Les
mesures des jeux synthétiques E1/E2/E3, E4 et development_v3 servent au développement; elles ne
valident pas une généralisation indépendante ou un usage en production. E4 a passé son critère
préenregistré de non-régression et de gain sur certaines métriques, mais n'a pas amélioré
l'exactitude (2/18 dans les deux versions) et son verdict natif reste FAIL. Les limites et compteurs sont dans
[`answer-quality-development.md`](answer-quality-development.md).

## Modes de réponse

Le mode livré est `grounded-local-v3`. Son vecteur est l'embedding ONNX local
`paraphrase-multilingual-minilm-l12-v2-onnx-q@faf4aa4225822f3bc6376869cb1164e8e3feedd0`,
de dimension 384, calculé par l'adaptateur texte ONNX Runtime sur CPU. `embedding_model_id` définit l'espace
vectoriel : les embeddings feature-hashing historiques ne sont pas compatibles. La recherche
lexicale PostgreSQL est fusionnée avec le dense par reciprocal-rank fusion. Le moteur de réponse
retenu est déterministe et sourcé ; il ne doit pas être présenté comme un LLM complet et ne
nécessite aucune clé.

Deux autres stratégies locales ont été évaluées de façon bornée : un modèle NLI multilingue ONNX
et Qwen 2.5 0.5B quantifié avec sortie JSON contrainte. Elles ne sont pas utilisées par le runtime
retenu. L'identité, la licence, la taille et les empreintes des trois modèles locaux sont fixées dans
`infra/models/`; les résultats comparatifs et la décision sont dans l'ADR-0006.

`ProviderBundle` regroupe les protocoles `EmbeddingProvider` et `AnswerProvider`. Le registre refuse
tout mode inconnu ; un fournisseur local ou compatible OpenAI peut être ajouté à ces deux frontières
avec un mode et un coût explicites. Aucun appel externe ni coût n’est effectué par défaut.

## Accès et observabilité

Les mots de passe sont vérifiés côté API et stockés sous forme de hash Argon2. Les jetons JWT sont courts (30 minutes par défaut). Les rôles sont appliqués au niveau de chaque route : administrateur (toutes actions), analyste (import, lecture, questions, extraction) et lecteur (lecture, questions, extraction). L’audit enregistre acteur, action, résultat, document éventuel, date et identifiant de corrélation.

Le middleware renvoie `X-Correlation-ID` et produit des logs JSON sans corps de document, mot de passe ou token. Prometheus est exposé sous `/metrics`; `/health` vérifie le processus et `/ready` vérifie PostgreSQL et Redis. `/api/v1/status` est réservé à l’administrateur.

Un second middleware applique une fenêtre glissante aux routes d'authentification, question et
import. Il utilise l'adresse client ASGI, ignore les en-têtes transférés non fiables, borne son état
à 10 000 clés et renvoie `429` avec `Retry-After`. Il est volontairement mono-processus : un service
horizontal exigerait Redis ou un edge partagé avec une politique d'adresse proxy explicitement
configurée.

## Déploiement local

`compose.yaml` compose PostgreSQL/pgvector, Redis, migration, seed synthétique, API, worker et web. Les ports publiés sont liés à l’interface loopback ; le volume `documents_data` est partagé seulement entre API et worker. Les images externes et les actions CI sont épinglées à des révisions immuables. Chaque service basé sur l’image locale API déclare le même build et `pull_policy: build`, afin qu’un lancement ciblé normal reconstruise avant exécution. `scripts/check_supply_chain_refs.py` contrôle ces invariants. Nginx limite les requêtes à 11 MiB, ajoute `nosniff`, `DENY` contre l’iframe, `no-referrer` et une CSP restrictive.

`compose.public-demo.yaml` superpose un profil synthétique avec un seul compte analyste, secrets
runtime obligatoires, quotas plus stricts et `/metrics` masqué. Les ports restent sur loopback : cet
overlay prépare une démo sûre, pas une publication Internet. Ces contrôles ne remplacent pas TLS,
un reverse proxy opéré, un quota distribué, la rotation des secrets, une politique IAM S3 ou une
vérification de provenance des images.
