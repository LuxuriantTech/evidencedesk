# EvidenceDesk — preuve carrière

## Résumé en 30 secondes

J'ai construit une application full-stack de revue documentaire pour un dossier fournisseur
entièrement synthétique. React appelle une API FastAPI sécurisée par rôles ; les imports passent par
Redis et un worker ARQ, puis PostgreSQL/pgvector conserve les passages, extractions et audits. Le
mode sans clé répond de façon extractive avec document, page et extrait, ou s'abstient. J'ai aussi
archivé deux jeux holdout distincts avec hashes et locks locaux : ils ont échoué, ce qui montre
précisément où la méthode lexicale ne généralise pas encore. Ces contrôles ne valent pas scellement
externe.

## Problème résolu

Une équipe opérations doit retrouver rapidement une date, un montant ou une obligation sans accepter
une réponse impossible à vérifier. EvidenceDesk relie chaque réponse et champ extrait au passage
masqué utilisé, conserve un audit corrélé et refuse les questions sans preuve suffisante.

## Architecture défendable

- React/TypeScript et nginx pour l'interface ;
- API REST FastAPI, JWT court, Argon2 et RBAC `admin`/`analyst`/`reader` ;
- PostgreSQL 16 avec pgvector HNSW et recherche textuelle GIN ;
- Redis/ARQ pour les tâches, trois tentatives et idempotence par SHA-256 ;
- stockage local derrière un protocole substituable ;
- logs JSON, UUID de corrélation et métriques Prometheus ;
- Docker Compose et CI préparée, tests pytest/Vitest/Playwright/axe.

## Difficultés rencontrées

1. Une suppression pouvait courir contre le worker : API et worker verrouillent maintenant tâche
   puis document, et un test concurrent prouve que les chunks/extractions ne survivent pas.
2. Une erreur d'embedding, d'extraction ou d'insertion pouvait laisser `processing` : des tests
   injectent désormais ces pannes et vérifient reprise puis échec terminal.
3. Le premier E2E simulait l'API : un job distinct démarre maintenant la pile et vérifie réellement
   navigateur → API → Redis/worker → PostgreSQL.
4. axe-core a trouvé des listes de définitions invalides et des zones scrollables non focalisables ;
   la structure HTML et l'accès clavier ont été corrigés.
5. Le holdout a falsifié l'hypothèse de généralisation ; v2 et v3 sont conservés et déclarés
   interdits de réutilisation pour l'optimisation.
6. La limite fichier dans le handler ne bornait pas le spool multipart : un middleware ASGI coupe
   maintenant le corps complet avant le parseur, y compris sans `Content-Length`.

## Compromis techniques

- Le feature hashing est rapide, déterministe et sans clé, mais il n'est pas sémantique et généralise
  mal aux synonymes/domaines nouveaux.
- La réponse extractive limite l'hallucination et le coût, mais ne synthétise pas plusieurs passages
  comme un LLM.
- Le stockage local rend la démo simple ; le protocole prépare S3 sans prétendre qu'un adaptateur S3
  a été testé.
- Les identifiants de démo sont publics et acceptables localement seulement.
- Les budgets bornent pages, caractères et chunks ; le parsing PDF isolé a un timeout et des
  limites POSIX, mais la limite mémoire dure n'existe pas sous Windows natif.

## Résultats mesurés

- corpus versionné : 40 cas, dont 25 répondables, 10 sans réponse et 5 ambigus/adversariaux ;
- développement v1.2 : 100 % sur les trois portes, utile seulement pour le développement ;
- holdout v2 : citations 70 %, abstention 80 %, extraction F1 94,12 %, verdict FAIL ;
- holdout v3 : matcher citations 50 %, abstention 70 %, extraction F1 33,33 %, verdict FAIL ;
- audit citation v3 à égalité littérale : 0/8, révélant une métrique historique trop permissive ;
- parcours Playwright réel : import `202 queued` jusqu'à `completed`, réponse/citation, abstention,
  extraction, PII, audit, évaluation et suppression ;
- axe-core : aucune violation `critical` ou `serious` sur ce parcours desktop et mobile ;
- coût fournisseur mesuré : 0 USD, car aucun fournisseur externe n'est appelé.

Les sorties exactes sont dans `artifacts/evaluations/`, `artifacts/coverage.json` et les journaux de
tests locaux. Les latences de l'évaluateur sont en mémoire et ne sont pas des latences API.

## Limites à dire spontanément

- prototype local, aucun client ni usage production ;
- objectifs holdout non atteints ;
- pas d'embedding sémantique pré-entraîné ni de LLM ;
- OCR, PDF image, tableaux complexes et chiffrement au repos hors périmètre ;
- pas de limite mémoire PDF dure sous Windows natif ni d'atomicité fichier/transaction DB ;
- CI GitHub préparée mais non exécutée tant que le dépôt n'est pas publié ;
- pas de benchmark de charge ni de validation Windows native.

## Points CV proposés — anglais

- Built a Docker Compose document-review application with FastAPI, React/TypeScript,
  PostgreSQL/pgvector and Redis/ARQ, including server-side RBAC, page-level citations, PII redaction
  and correlated audit logs.
- Implemented a versioned 40-case evaluation protocol with local run locks, dataset hashes and
  engine fingerprints; retained failed holdout results and documented that the controls are not an
  external proof of one-shot execution.
- Verified a real browser-to-worker workflow with Playwright and axe-core, covering asynchronous
  ingestion, sourced answers, abstention, extraction, audit and controlled deletion with no critical
  or serious accessibility finding on the tested desktop/mobile path.

## Points CV proposés — français

- Développement d'une application Docker Compose de revue documentaire avec FastAPI,
  React/TypeScript, PostgreSQL/pgvector et Redis/ARQ, incluant RBAC serveur, citations par page,
  masquage PII et audit corrélé.
- Mise en place d'un protocole versionné de 40 cas avec locks locaux, empreintes des datasets et
  fingerprint moteur ; conservation des résultats holdout négatifs et limites de scellement
  documentées.
- Validation d'un parcours réel navigateur–worker avec Playwright et axe-core : ingestion
  asynchrone, réponse sourcée, abstention, extraction, audit et suppression contrôlée, sans finding
  d'accessibilité critique ou sérieux sur le parcours desktop/mobile testé.

## Questions probables d'entretien

### Pourquoi pgvector si le mode n'est pas vraiment sémantique ?

Le schéma, les index et la requête vectorielle prouvent l'intégration. Le vecteur actuel est un
baseline déterministe sans clé ; le holdout montre qu'il ne faut pas le vendre comme un embedding
sémantique. La prochaine version remplacerait le fournisseur derrière l'interface, puis utiliserait
un nouveau holdout.

### Comment empêchez-vous une instruction malveillante dans un document ?

Le document est traité comme donnée non fiable. Le fournisseur extractif ne possède aucun outil et
refuse les requêtes d'injection/élévation reconnues ; surtout, aucun passage ne peut modifier les
rôles ou la politique serveur.

### Pourquoi conserver un FAIL dans un portfolio ?

Parce qu'un système RAG ne se juge pas sur une démo choisie. Le produit fonctionne, mais le holdout
falsifie la généralisation. Conserver le FAIL, le lock et les hashes rend la démarche vérifiable.

### L'évaluation couvre-t-elle le système complet ?

Non. Elle appelle le moteur en mémoire. Le parcours complet est vérifié séparément par Playwright ;
il ne faut pas confondre exactitude documentaire et intégration système.

### Que feriez-vous ensuite ?

Corriger la définition stricte des citations, introduire un vrai embedding local versionné, élargir
l'extraction, geler les paramètres, puis créer un holdout v4 indépendant. Les v2/v3 resteraient
historiques et ne serviraient plus au réglage.

## Cohérence scolaire à corriger avant publication

Le CV PDF le plus récent, daté localement du 2026-08-24, indique : projet de BSc (Honours) Computer
Science with Artificial Intelligence à **The Open University**, choix de cursus confirmé et
inscription aux modules en attente. Le README de profil GitHub local daté du 2026-08-21 affirme un
démarrage à **University of London** en octobre 2026. Cette ligne est obsolète par rapport au CV le
plus récent et ne doit pas présenter le cursus comme commencé.

Texte prudent proposé :

> Planning a BSc (Honours) in Computer Science with Artificial Intelligence at The Open University;
> course choice confirmed, module registration pending.

Ne publier cette correction qu'après validation par Ardian et sans transformer une intention ou une
inscription en attente en cursus commencé.
