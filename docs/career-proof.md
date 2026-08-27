# EvidenceDesk — preuve carrière

## Résumé en 30 secondes

J'ai construit localement une application full-stack de revue documentaire pour un dossier
fournisseur entièrement synthétique. React appelle une API FastAPI avec RBAC ; les imports passent
par Redis/ARQ, puis PostgreSQL/pgvector conserve passages, extractions et audits. Le moteur actif
utilise un embedding ONNX local et une recherche hybride, sans clé ni appel externe ; la réponse
reste extractive, sourcée ou abstentionniste. Son holdout v7 indépendant a échoué : c'est un
prototype technique et non une preuve de qualité RAG généralisable.

## Architecture et choix défendables

- React/TypeScript et nginx ; API REST FastAPI, JWT court, Argon2 et RBAC
  `admin`/`analyst`/`reader` ;
- PostgreSQL 16 avec pgvector HNSW et recherche textuelle GIN ; Redis/ARQ pour les tâches,
  trois tentatives et l'idempotence par SHA-256 ;
- stockage local derrière un protocole substituable, logs JSON, UUID de corrélation et métriques
  Prometheus ;
- modèle local : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, export ONNX Qdrant
  figé, 384 dimensions, licence Apache-2.0 ; le code du projet est sous licence MIT ;
- recherche hybride : fusion de rangs entre candidats lexicaux et denses. Le choix évite de traiter
  des scores bruts hétérogènes comme s'ils étaient calibrés.

## Difficultés et compromis

1. Une suppression pouvait courir contre le worker : API et worker verrouillent désormais tâche puis
   document ; un test concurrent couvre la disparition des chunks et extractions.
2. Les échecs d'embedding, extraction ou insertion pouvaient laisser `processing` : les tests
   injectent ces pannes et vérifient reprise ou échec terminal.
3. Un E2E simulé ne suffisait pas : le parcours local couvre navigateur → API → Redis/worker →
   PostgreSQL.
4. La réponse extractive, l'abstention et les citations réduisent le risque d'invention, mais ne
   produisent pas de synthèse multi-passage comme un LLM.
5. CPU local et aucun fournisseur payant simplifient confidentialité et coût, au prix d'un modèle de
   384 dimensions, d'un téléchargement mesuré à 266 906 689 octets et de performances à démontrer
   hors corpus de test.

## Résultat réellement mesuré

Le holdout v7 est l'artefact de qualité à citer, pas la démo ni le développement : 40 cas,
`deterministic-evidence-v3` avec récupération `hybrid`, sur CPU (`12th Gen Intel Core i5-12600KF`,
16 CPU logiques, 16 768 458 752 octets de RAM, ONNX Runtime `CPUExecutionProvider`). Verdict :
**FAIL**.

- précision/rappel de citation : 80 % / 48 % (12 correctes sur 15 retournées ; 12/25 preuves gold) ;
- cas répondables corrects : 36 % (9/25) ;
- abstention correcte : 80 % (12/15) ;
- F1 d'extraction : 45,67 % ; Recall@5 : 100 % (25/25) ;
- taux d'erreur : 0 % ; coût API externe : 0 USD.

Ces chiffres montrent que retrouver une preuve dans les cinq premiers candidats ne suffit pas à
produire une réponse/citation/extraction correcte. Onze cas répondables ont été refusés et un cas
adversarial a suivi une instruction injectée. Les v4 et v5 se sont arrêtés au préflight, sans
métrique de qualité, et ne sont pas rejoués. Les résultats de développement servent au réglage, pas
à établir une qualité générale.

## Limites à dire spontanément

- prototype local ; aucun client, usage production, SLA, URL publique ou CI GitHub exécutée ;
- holdout v7 en échec : aucun objectif de qualité ne doit être présenté comme atteint ;
- les locks et hashes locaux améliorent la traçabilité mais ne prouvent pas un scellement externe ni
  l'absence de consultation ;
- OCR, PDF image, tableaux complexes, chiffrement applicatif, charge concurrente et validation
  Windows native restent hors périmètre ;
- le benchmark du moteur ne remplace pas une validation de toute la pile, et le parcours E2E ne
  remplace pas une mesure de qualité documentaire.

## Réponses d'entretien

### Pourquoi pgvector et un embedding ONNX local ?

Pour exercer une chaîne dense réellement locale, vérifiée par manifest et hash, sans clé runtime.
La récupération hybride conserve aussi le lexical. Le holdout v7 échoue toutefois : l'intégration
technique est démontrée, pas l'efficacité finale du système.

### Pourquoi garder un résultat FAIL ?

Parce qu'une démo choisie ne mesure pas la généralisation. Le FAIL v7, les compteurs bruts et les
artefacts permettent d'expliquer précisément la limite : Recall@5 parfait, mais réponse, citation
et extraction insuffisantes.

### Quel serait le prochain travail ?

Créer un nouveau développement séparé, améliorer la résistance aux injections, la décision et le
schéma d'extraction, puis geler un moteur avant un nouveau holdout indépendant. Le v7 déjà ouvert
ne doit pas devenir un jeu de réglage.

### L'évaluation couvre-t-elle tout le produit ?

Non. L'évaluateur mesure le moteur ; Playwright couvre séparément le flux réel. Ce sont deux
preuves complémentaires, aucune ne valide à elle seule la qualité de production.

## Points CV proposés — anglais

- Built a local document-review prototype with FastAPI, React/TypeScript, PostgreSQL/pgvector and
  Redis/ARQ, including server-side RBAC, page-level evidence, PII redaction and correlated audit logs.
- Implemented a local, CPU-only ONNX embedding pipeline and hybrid retrieval with pinned model
  identity, file hashes and a 40-case evaluation artifact; reported the v7 holdout FAIL rather than
  presenting development or demo results as general quality.
- Verified a real browser-to-worker workflow with Playwright, covering asynchronous ingestion,
  sourced answers, abstention, extraction, audit and controlled deletion.

## Points CV proposés — français

- Développement d'un prototype local de revue documentaire avec FastAPI, React/TypeScript,
  PostgreSQL/pgvector et Redis/ARQ, incluant RBAC serveur, preuves par page, masquage PII et audit
  corrélé.
- Mise en place d'un pipeline d'embedding ONNX local sur CPU et de recherche hybride, avec identité
  modèle figée, hashes de fichiers et artefact d'évaluation de 40 cas ; conservation du FAIL holdout
  v7 sans présenter la démo ou le développement comme preuve de qualité générale.
- Vérification d'un parcours réel navigateur–worker avec Playwright : ingestion asynchrone, réponse
  sourcée, abstention, extraction, audit et suppression contrôlée.

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
