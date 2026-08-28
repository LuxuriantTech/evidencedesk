# Spec: EvidenceDesk v1

**Author:** Codex, sous la responsabilité d’Ardian Mehaj
**Date:** 2026-08-26
**Status:** Approved; exigences et autorisation locale fournies par le propriétaire le 2026-08-26
**Reviewers:** Ardian Mehaj (acceptation fonctionnelle par la demande source), revue adversariale indépendante avant livraison

## Context

EvidenceDesk est un assistant de traitement de dossiers fournisseurs synthétiques pour les équipes
opérationnelles. Il doit transformer des PDF texte, fichiers TXT et Markdown en informations
structurées, puis répondre à des questions uniquement lorsque des passages précis du corpus les
étayent. La démonstration doit rester reproductible sans clé externe et ne doit pas se présenter
comme un service utilisé par de vrais clients.

Le produit sert aussi de preuve de compétences full-stack : FastAPI/Python typé, React/TypeScript,
PostgreSQL/pgvector, Redis et worker asynchrone, Docker Compose, tests, CI, sécurité,
observabilité et évaluation reproductible. Le corpus public est entièrement synthétique ; toute
instruction apparaissant dans un document est une donnée non fiable et n’a aucune autorité sur le
système.

Le parcours prioritaire tient en moins de trois minutes : connexion, ouverture d’un dossier,
observation du traitement, question répondable avec citation, question impossible avec abstention,
extraction structurée et preuves, masquage PII, audit, puis résultats d’évaluation calculés.

### Architecture cible

```text
React/TypeScript -> REST FastAPI -> PostgreSQL 16 + pgvector
                         |       -> stockage local (adaptateur S3 prévu)
                         +-----> Redis -> worker ARQ -> extraction/indexation
                         +-----> métriques Prometheus + logs JSON corrélés
```

Le mode par défaut utilise des embeddings déterministes locaux et une réponse extractive. Un
contrat de fournisseur permet l’ajout ultérieur d’un modèle local ou compatible OpenAI, mais v1
n’effectue aucun appel externe et n’exige aucune clé.

## Functional Requirements

- FR-1: Le système MUST authentifier les utilisateurs avec mot de passe Argon2id et jeton signé à durée limitée.
- FR-2: Le serveur MUST appliquer les rôles `admin`, `analyst` et `reader` sur chaque action protégée ; le frontend MUST NOT être l’unique contrôle d’autorisation.
- FR-3: Le démarrage de démonstration MUST créer des comptes documentés sans enregistrer leur mot de passe en clair en base.
- FR-4: L’API MUST accepter les PDF texte, TXT et Markdown, vérifier extension, signature/MIME, taille et contenu minimal, puis utiliser un nom de stockage généré.
- FR-5: Le stockage MUST passer par une interface avec implémentation locale ; une interface compatible objet/S3 SHOULD être substituable sans modifier les services métier.
- FR-6: Chaque import MUST créer un document et une tâche corrélés, visibles dans les états `queued`, `processing`, `completed` ou `failed`.
- FR-7: Le worker MUST retraiter une erreur transitoire au plus trois fois et MUST conserver l’erreur publique assainie ; une clé d’idempotence de contenu MUST empêcher un traitement involontaire en double.
- FR-8: L’extraction MUST préserver page, section, ordre et texte source ; le découpage SHOULD respecter les paragraphes et titres avant une limite de taille.
- FR-9: Les passages MUST être stockés dans PostgreSQL avec un vecteur pgvector et un index de recherche textuelle ; la recherche MUST combiner sémantique locale et correspondance lexicale avec paramètres versionnés.
- FR-10: Une réponse MUST contenir statut, texte extractif, niveau de confiance et au moins une citation avec document, page, section et extrait exact ; l’extrait MUST être une sous-chaîne du passage stocké.
- FR-11: Le système MUST s’abstenir explicitement lorsque la preuve est insuffisante ou contradictoire et MUST NOT transformer une ambiguïté en certitude.
- FR-12: Les instructions trouvées dans un document MUST être traitées comme contenu non fiable ; elles MUST NOT modifier le rôle, les outils, les politiques ou la logique de réponse.
- FR-13: L’extraction structurée MUST produire organisation, type de document, dates d’effet et de renouvellement, montants, obligations, responsables, risques/incohérences et citations par champ.
- FR-14: Les e-mails, téléphones et identifiants personnels synthétiques reconnus MUST être masqués dans le texte indexé, les réponses et l’interface ; les logs MUST NOT contenir le document, mot de passe, jeton ou PII brute.
- FR-15: Un administrateur MUST pouvoir supprimer un document de façon contrôlée, y compris fichier, passages et extractions, tout en conservant une trace d’audit sans contenu documentaire.
- FR-16: Le système MUST journaliser acteur, action, document, date UTC, résultat et identifiant de corrélation pour connexion, import, traitement, question, extraction et suppression.
- FR-17: L’évaluateur MUST exécuter un jeu versionné d’au moins 40 cas : au moins 25 répondables, 10 sans réponse et 5 ambigus/adversariaux, séparés en développement et holdout.
- FR-18: L’évaluation MUST calculer précision des citations, exactitude/F1 d’extraction, exactitude d’abstention, médiane et p95 de latence, taux d’erreur, coût estimé et mode d’exécution.
- FR-19: Le holdout MUST être ouvert une seule fois pour la version de paramètres scellée ; toute optimisation ultérieure MUST utiliser un nouveau holdout indépendant.
- FR-20: Le mode `extractive-local` MUST fonctionner dans les tests, la CI et la démonstration sans clé ; chaque autre fournisseur MUST exposer explicitement son mode et son coût.
- FR-21: L’interface MUST proposer connexion, tableau de bord, dossier, suivi des tâches, vue document/réponse côte à côte, extraction, audit, évaluation et état administratif.
- FR-22: Les citations MUST être activables au clavier et faire défiler la vue document jusqu’au passage/page correspondant.
- FR-23: L’API MUST exposer santé, disponibilité et métriques ; traitements et recherches MUST émettre durée, succès/échec et corrélation dans des logs JSON structurés.
- FR-24: En mode démonstration publique, l’import MUST exiger une attestation synthétique, limiter le contenu à une allowlist SHA-256 versionnée et afficher un avertissement d’absence de données réelles.
- FR-25: Docker Compose MUST démarrer web, API, worker, PostgreSQL/pgvector et Redis avec données de démonstration reproductibles.
- FR-26: Le dépôt MUST inclure script de démonstration, exemples API, règles de conservation, architecture, sécurité, évaluation, ADR, licence, contribution, signalement sécurité et préparation carrière fondée sur les mesures.

## Non-Functional Requirements

- NFR-1: Les composants métier critiques MUST atteindre au moins 80 % de couverture de branches ou lignes, mesurée et publiée par la CI.
- NFR-2: Les tests unitaires, API, intégration, worker, autorisation, idempotence, reprise, citation, abstention, PII et Playwright MUST passer avant livraison.
- NFR-3: Ruff, mypy strict, ESLint et `tsc --noEmit` MUST passer sans erreur.
- NFR-4: Le parcours principal MUST avoir zéro violation axe-core `critical` ou `serious` dans le test E2E.
- NFR-5: Sur le corpus de démonstration local, la recherche SHOULD répondre avec une latence p95 inférieure à 750 ms hors première initialisation, mesurée par l’évaluateur.
- NFR-6: Le corps HTTP MUST être limité avant parsing multipart, les fichiers importés MUST être limités à 10 MiB, les questions à 1 000 caractères et le traitement MUST borner pages, caractères extraits et chunks avec des valeurs configurables.
- NFR-7: Les jetons MUST expirer au plus tard après 30 minutes et les réponses d’erreur MUST NOT divulguer de trace interne.
- NFR-8: Les services Docker MUST avoir des contrôles de santé et l’API MUST rester indisponible tant que ses dépendances obligatoires ne répondent pas.
- NFR-9: Le build et la démonstration MUST être reproductibles sous Linux/WSL ; la procédure Windows Docker Desktop MUST être vérifiée par commandes compatibles, sans prétendre à un test hôte non effectué.
- NFR-10: Les scans de secrets et dépendances MUST ne signaler aucune fuite ni vulnérabilité critique connue ; toute vulnérabilité importante restante MUST être documentée.
- NFR-11: Les objectifs empiriques holdout sont citation >= 90 %, abstention >= 85 %, extraction >= 90 % ; un échec MUST être rapporté sans modification silencieuse des critères.
- NFR-12: L’interface MUST rester utilisable à 375 px et 1280 px, avec focus visible, contrastes WCAG AA et navigation complète au clavier sur le parcours principal.

## Acceptance Criteria

### AC-1: Connexion et rôles côté serveur (FR-1, FR-2, FR-3)
Given trois comptes de démonstration aux rôles distincts
When chacun se connecte et tente lecture, import, consultation d’audit et suppression
Then l’API autorise exactement la matrice documentée et renvoie 401 ou 403 pour chaque action interdite

### AC-2: Validation défensive des fichiers (FR-4, FR-5, NFR-6)
Given des PDF/TXT/Markdown valides et des fichiers faux, trop grands, vides ou au nom traversant
When un analyste les importe
Then seuls les formats valides sous 10 MiB sont stockés sous un identifiant généré via l’adaptateur configuré et les autres reçoivent une erreur 4xx stable

### AC-3: Traitement asynchrone observable (FR-6, FR-7)
Given un document valide importé
When le worker traite sa tâche
Then les états passent de queued à processing puis completed avec corrélations document/tâche et durée mesurée

### AC-4: Idempotence et reprise (FR-7)
Given deux imports concurrents du même contenu et une première tentative transitoirement défaillante
When les tâches sont consommées
Then un seul index documentaire est produit et la tâche réussit en trois tentatives maximum sans passages dupliqués

### AC-5: Pages, sections et pgvector (FR-8, FR-9)
Given un dossier synthétique traité
When les passages sont inspectés dans PostgreSQL
Then chaque passage possède document, page, section, ordre, texte masqué, tsvector et vecteur pgvector de dimension versionnée

### AC-6: Réponse avec preuve exacte (FR-10)
Given une question répondable du jeu d’évaluation
When un lecteur interroge le dossier
Then la réponse extractive cite document, page, section et un extrait exact présent dans le passage enregistré

### AC-7: Abstention correcte (FR-11)
Given une question sans réponse dans le corpus
When le seuil scellé est appliqué
Then la réponse porte le statut `abstained`, explique l’insuffisance et ne fournit aucune affirmation factuelle non sourcée

### AC-8: Ambiguïté et injection documentaire (FR-11, FR-12)
Given des dates contradictoires ou un passage demandant d’ignorer les règles du système
When le dossier est interrogé
Then le système signale l’ambiguïté ou s’abstient et traite l’instruction comme une citation non fiable sans l’exécuter

### AC-9: Extraction structurée traçable (FR-13)
Given un contrat fournisseur synthétique traité
When l’extraction est consultée
Then chaque valeur non nulle expose au moins une citation exacte et les incohérences conservent les deux preuves concernées

### AC-10: Masquage PII (FR-14)
Given des e-mails, téléphones et identifiants synthétiques dans un document
When le document est traité, recherché et observé dans les logs
Then l’interface et l’index affichent des marqueurs masqués et aucune PII brute n’apparaît dans les logs

### AC-11: Suppression contrôlée (FR-15, FR-16)
Given un document traité
When un administrateur confirme sa suppression
Then le fichier, ses passages et extractions disparaissent, le document devient supprimé et une trace d’audit minimale demeure

### AC-12: Audit complet et filtrable (FR-16)
Given les actions du scénario de démonstration
When l’administrateur ouvre le journal
Then chaque action affiche acteur, action, document éventuel, UTC, résultat et corrélation sans contenu sensible

### AC-13: Protocole de 40 cas (FR-17, FR-19)
Given le manifeste d’évaluation versionné
When le validateur de dataset est lancé
Then il confirme au moins 40 cas, 25 répondables, 10 non répondables, 5 adversariaux et des ensembles development/holdout disjoints

### AC-14: Métriques réellement calculées (FR-18, NFR-11)
Given une version de corpus, paramètres et graine scellés
When l’évaluation est exécutée une fois sur le holdout
Then un artefact horodaté contient numérateurs, dénominateurs, latences, erreurs, coût nul en mode extractif et verdict contre les seuils préenregistrés

### AC-15: Fonctionnement sans clé (FR-20)
Given un environnement sans clé LLM
When Docker Compose, les tests et l’évaluateur sont lancés
Then le système fonctionne en `extractive-local` et n’émet aucune requête vers un fournisseur de modèle

### AC-16: Parcours UI accessible et responsive (FR-21, FR-22, NFR-4, NFR-12)
Given les écrans 375 px et 1280 px
When le parcours principal est réalisé uniquement au clavier et contrôlé avec axe
Then navigation, états, erreurs, citations et focus sont utilisables sans violation critical ou serious

### AC-17: Santé et observabilité (FR-23, NFR-8)
Given la pile complète puis une dépendance indisponible
When santé, disponibilité et métriques sont interrogées
Then santé décrit le processus, disponibilité reflète les dépendances et les compteurs/durées sont exposés sans données sensibles

### AC-18: Limite synthétique publique (FR-24)
Given le mode public activé
When un import ne porte pas l’attestation synthétique
Then l’API le refuse et l’interface explique que seules des données entièrement synthétiques sont autorisées

### AC-19: Démarrage reproductible (FR-25, NFR-9)
Given un clone propre avec Docker et Compose
When la commande documentée de démarrage et le seed de démo sont lancés
Then les cinq services deviennent sains et le compte de démo ouvre le dossier synthétique

### AC-20: Livraison documentaire vérifiable (FR-26, NFR-10)
Given le dépôt prêt au handoff
When les liens, exemples, scans, captures et commandes documentées sont vérifiés
Then chaque artefact existe, correspond au produit réel et aucune clé, donnée personnelle ou métrique inventée n’est présente

## Edge Cases

- EC-1: Un fichier nommé `../../contract.pdf` -> le nom utilisateur est conservé comme métadonnée assainie et jamais utilisé comme chemin.
- EC-2: Un fichier `.pdf` contenant du HTML -> l’import est refusé pour signature/type incohérents.
- EC-3: Un PDF chiffré, scanné ou sans texte -> le traitement échoue proprement avec code public documenté et possibilité de suppression.
- EC-4: Redis est indisponible pendant l’import -> aucun faux état completed ; l’API retourne 503 et l’audit enregistre l’échec.
- EC-5: Le worker s’arrête après passage à processing -> la tâche réessayée reprend idempotemment sans passages dupliqués.
- EC-6: Deux utilisateurs importent le même contenu -> l’idempotence est limitée au dossier/organisation et ne divulgue aucun document d’un autre périmètre.
- EC-7: Une citation pointe vers un texte masqué -> l’extrait et le texte affiché utilisent exactement la même représentation masquée.
- EC-8: Deux sources plausibles se contredisent -> la réponse est ambiguous ou abstained et cite les deux sources.
- EC-9: La question contient HTML, SQL ou instruction de prompt -> elle reste une chaîne bornée et ne modifie ni requête SQL ni politique.
- EC-10: Un lecteur appelle directement DELETE ou POST upload -> le serveur répond 403 et écrit un audit d’autorisation refusée.
- EC-11: Le fichier local a disparu avant suppression -> la suppression DB reste contrôlée, l’anomalie est journalisée et la réponse ne divulgue pas le chemin.
- EC-12: Aucun résultat d’évaluation n’existe -> la page affiche un état vide et MUST NOT afficher des valeurs fictives.

## API Contracts

```typescript
type Role = "admin" | "analyst" | "reader";
type DocumentStatus = "queued" | "processing" | "completed" | "failed" | "deleted";
type AnswerStatus = "answered" | "abstained" | "ambiguous";

interface TokenRequest { username: string; password: string }
interface UserView { id: string; username: string; role: Role }
interface TokenResponse { access_token: string; token_type: "bearer"; expires_in: number; user: UserView }
interface ApiError { code: string; message: string; correlation_id: string }

interface DocumentView {
  id: string; dossier_id: string; filename: string; media_type: string;
  status: DocumentStatus; task_id: string | null; correlation_id: string;
  error_code: string | null; created_at: string; completed_at: string | null;
}

interface Citation {
  chunk_id: string; document_id: string; document_name: string;
  page: number; section: string | null; excerpt: string;
}
interface AskRequest { question: string }
interface AskResponse {
  status: AnswerStatus; answer: string; confidence: number;
  mode: "extractive-local"; citations: Citation[]; correlation_id: string;
}
interface EvidenceValue<T> { value: T | null; citations: Citation[] }
interface ExtractionView {
  organization_name: EvidenceValue<string>; document_type: EvidenceValue<string>;
  effective_date: EvidenceValue<string>; renewal_date: EvidenceValue<string>;
  important_amounts: EvidenceValue<string[]>; obligations: EvidenceValue<string[]>;
  responsible_people: EvidenceValue<string[]>; risks: EvidenceValue<string[]>;
}
interface EvaluationRun {
  id: string; dataset_version: string; parameters_version: string; split: "development" | "holdout";
  mode: string; citation_precision: number; extraction_score: number;
  abstention_accuracy: number; latency_median_ms: number; latency_p95_ms: number;
  error_rate: number; estimated_cost_usd: number; verdict: "PASS" | "FAIL";
}
```

Endpoints v1 :

- `POST /api/v1/auth/token` -> `TokenResponse` ou 401.
- `GET /api/v1/auth/me` -> `UserView`.
- `GET /api/v1/dossiers` et `GET /api/v1/dossiers/{id}` -> dossiers autorisés.
- `POST /api/v1/dossiers/{id}/documents` multipart (`file`, `is_synthetic`) -> 202 `DocumentView`.
- `GET /api/v1/documents/{id}` -> `DocumentView` ; `DELETE /api/v1/documents/{id}` -> 204.
- `GET /api/v1/documents/{id}/content` -> pages masquées et sections.
- `POST /api/v1/dossiers/{id}/ask` + `AskRequest` -> `AskResponse`.
- `GET /api/v1/dossiers/{id}/extraction` -> `ExtractionView`.
- `GET /api/v1/audit-events` -> page filtrable d’événements.
- `GET /api/v1/evaluations` et `POST /api/v1/evaluations/run` -> résultats calculés, exécution admin.
- `GET /health`, `GET /ready`, `GET /metrics` -> état opérationnel.

## Data Models

| Entity | Field | Type | Constraints |
|---|---|---|---|
| User | id | UUID | primary key |
| User | username | text | unique, normalized |
| User | password_hash | text | Argon2id uniquement |
| User | role | enum | admin, analyst, reader |
| Dossier | id | UUID | primary key |
| Dossier | name | text | non vide |
| Document | id | UUID | primary key |
| Document | dossier_id | UUID | foreign key, cascade contrôlée |
| Document | content_sha256 | char(64) | unique avec dossier et état actif |
| Document | storage_key | text | générée côté serveur |
| Document | status | enum | queued, processing, completed, failed, deleted |
| Document | correlation_id | UUID | non nul, indexé |
| ProcessingTask | id | UUID | primary key et corrélation worker |
| ProcessingTask | document_id | UUID | unique par génération de contenu |
| ProcessingTask | attempts | integer | 0..3 |
| Chunk | id | UUID | primary key |
| Chunk | document_id/page/ordinal | UUID/int/int | contrainte unique |
| Chunk | section/text | text/text | texte masqué non vide |
| Chunk | embedding | vector(384) | modèle local versionné |
| Chunk | search_vector | tsvector | index GIN |
| Extraction | id | UUID | primary key |
| Extraction | document_id/schema_version | UUID/text | unique |
| Extraction | payload | jsonb | valeurs et références de chunks |
| AuditEvent | id | UUID | primary key |
| AuditEvent | actor/action/result | UUID/text/text | acteur nullable pour login inconnu |
| AuditEvent | document_id/correlation_id | UUID/UUID | document nullable, corrélation non nulle |
| EvaluationRun | id | UUID | primary key |
| EvaluationRun | dataset/parameters/mode | text/text/text | non nuls et versionnés |
| EvaluationRun | metrics | jsonb | numérateurs, dénominateurs et valeurs |

## Out of Scope

- OS-1: OCR, tableaux complexes, images et signatures manuscrites. La v1 accepte uniquement les PDF contenant du texte extractible.
- OS-2: Hébergement public, création GitHub, push, déploiement ou modification d’un profil. Ces opérations sont préparées localement puis soumises à un GO distinct.
- OS-3: Appels payants ou gestion de clés de fournisseur. Aucun appel externe en v1 ; le contrat d’extension est documenté.
- OS-4: Corpus réel, CV, contrats clients, données scolaires, trading ou pièces d’identité. Seules des données synthétiques versionnées sont autorisées pour la démo.
- OS-5: Témoignages, utilisateurs réels, SLA production ou promesse commerciale. Aucune adoption externe n’est revendiquée.
- OS-6: Édition collaborative et multi-tenant commercial. Un espace de démonstration unique avec contrôles de rôles suffit pour v1.
- OS-7: Correction automatique des documents ou prise de décision juridique. EvidenceDesk expose des preuves et des incohérences, il ne remplace pas un professionnel.
