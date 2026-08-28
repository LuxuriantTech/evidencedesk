# EvidenceDesk: preuve carrière

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
6. Le holdout v7 a restitué une instruction documentaire. Après le gel, j'ai séparé règles système,
   question, contenu non fiable et preuve, puis ajouté des gardes communs et des régressions
   adversariales. Le holdout n'a pas été rejoué : ce correctif logiciel ne devient pas une nouvelle
   métrique empirique.

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

- prototype local ; aucun client, usage production, SLA ou URL applicative publique ; la CI GitHub
  du dépôt source est exécutée séparément et ne constitue pas un déploiement ;
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

Le cycle expérimental est arrêté et aucun v8 ne doit être créé. Le travail de cette release candidate
porte sur sécurité, qualité, documentation et reproductibilité. Une future campagne scientifique,
si elle est décidée séparément, devra repartir d'un nouveau développement et d'un protocole gelé ;
v7 ne doit jamais devenir un jeu de réglage.

### L'évaluation couvre-t-elle tout le produit ?

Non. L'évaluateur mesure le moteur ; Playwright couvre séparément le flux réel. Ce sont deux
preuves complémentaires, aucune ne valide à elle seule la qualité de production.

## Points CV proposés en anglais

- Built a fully local document intelligence platform with asynchronous ingestion, hybrid retrieval,
  evidence-linked answers, server-side RBAC, PII masking and correlated audit logging.
- Created a reproducible blind evaluation framework that achieved 100% Recall@5 while identifying
  significant answer-generation, citation and extraction limitations on unseen document families.
- Validated the engineering workflow with 345 backend tests, 6 frontend unit tests, typed checks,
  real-stack Playwright coverage, dependency scanning and documented adversarial security checks;
  retained the negative v7 result instead of presenting demo data as general accuracy.

## Points CV proposés en français

- Développement d'une plateforme locale d'intelligence documentaire avec ingestion asynchrone,
  recherche hybride, réponses reliées aux preuves, RBAC serveur, masquage PII et audit corrélé.
- Création d'un cadre d'évaluation aveugle reproductible ayant obtenu 100 % de Recall@5 tout en
  révélant des limites importantes de réponse, citation et extraction sur des familles inédites.
- Validation du parcours d'ingénierie par 345 tests backend, 6 tests unitaires frontend, contrôles
  de typage, Playwright sur pile réelle, scans de dépendances et contrôles adversariaux documentés ;
  conservation du résultat v7 négatif au lieu de présenter la démo comme une précision générale.

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
