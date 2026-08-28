# Sécurité et données

## Périmètre de démonstration

Le mode `PUBLIC_DEMO_MODE=true` exige à la fois `is_synthetic=true` et un SHA-256 présent dans `datasets/public_demo_uploads.json`. Cette liste ne contient que quatre fichiers synthétiques versionnés dont les hashes sont vérifiés par test. Le chemin HTTP et le seed refusent tous deux un digest absent : un client ou un manifeste alternatif ne peut pas contourner la limite par une simple attestation. Pour importer un autre document synthétique en développement local, il faut désactiver explicitement ce mode ; aucun document réel ne doit être utilisé dans la démonstration.

## Modèle local et intégrité

L'embedding par défaut est téléchargé localement puis exécuté hors ligne sur CPU ; aucune question ni chunk n'est envoyé à un fournisseur par ce mode. Le manifest versionné fixe le dépôt Qdrant, sa révision et les SHA-256 de ses fichiers, dont `model_optimized.onnx`. Ces hashes détectent une divergence locale au contrôle, mais ne constituent ni signature de l'éditeur, ni preuve de provenance, ni protection contre un administrateur local qui remplace manifest et fichiers ensemble. Les évaluations et la démonstration documentées utilisent uniquement des données synthétiques ; aucune garantie de traitement approprié de données réelles n'en découle.

## Authentification et autorisation

- Les mots de passe sont hachés et vérifiés avec Argon2 ; ni mot de passe ni jeton n’est journalisé par l’application.
- Les JWT expirent après 30 minutes par défaut et sont contrôlés côté serveur avec l’utilisateur et son rôle courant.
- Administrateur : audit, statut et exécution d’évaluations en plus des droits analyste. Analyste : import, lecture, question et extraction. Lecteur : lecture, question et extraction.
- Le frontend améliore l’expérience, mais l’API est l’autorité de sécurité.

## Import, stockage et suppression

Les imports sont limités à PDF/TXT/Markdown et à 10 MiB de fichier. Avant que Starlette analyse ou spoule un multipart, un middleware ASGI coupe toute requête dépassant 11 MiB, avec ou sans `Content-Length` ; Nginx applique la même enveloppe. Une fenêtre glissante en mémoire limite aussi authentification, questions et imports par adresse cliente vue directement par l'ASGI, avec au plus 10 000 clés. Elle ne fait pas confiance à `X-Forwarded-For`, mais reste mono-processus : derrière le Nginx fourni, plusieurs visiteurs peuvent partager l'adresse du proxy, et une mise à l'échelle exige un quota partagé fiable. L’application valide ensuite type annoncé, signature et encodage. Le worker refuse par défaut plus de 200 pages, 1 000 000 de caractères extraits ou 5 000 chunks ; ces budgets sont configurables par `MAX_DOCUMENT_PAGES`, `MAX_EXTRACTED_CHARS` et `MAX_DOCUMENT_CHUNKS`. Les clés de stockage sont générées avec UUID ; une traversée de répertoire est refusée par résolution sous la racine locale.

La suppression via `DELETE /api/v1/documents/{document_id}` verrouille la tâche puis le document dans le même ordre que le worker, supprime le fichier local et les données dérivées, puis marque le document supprimé. Le worker revérifie ce marqueur sous verrou avant toute persistance et ses chemins d’échec ne peuvent pas le réactiver. Les métadonnées d’audit restent.

Le parsing PDF s’exécute dans un sous-processus `spawn` : le parent impose un délai mural et le child applique sous POSIX une limite d’espace d’adressage et de temps CPU. Les codes `pdf_processing_timeout` et `pdf_resource_limit` sont terminaux. La limite mémoire dure repose sur `resource` et n’est donc pas disponible sous Windows natif ; le parcours documenté et testé utilise Linux/WSL ou le conteneur Linux.

Limite résiduelle : la suppression du fichier et le commit PostgreSQL ne sont pas atomiques ; une panne DB entre les deux demanderait une réconciliation. La conservation est manuelle ; aucune purge planifiée, rétention chiffrée ou suppression cryptographique n’est implémentée. Pour toute donnée non synthétique, il faut définir une durée, des sauvegardes, une rétention des journaux et une purge vérifiable avant usage.

## PII, logs et prompt injection

Le worker remplace dans les chunks les e-mails, numéros de téléphone et identifiants synthétiques `SYN-ID-*`. C’est une démonstration de masquage par motifs, pas une détection complète de données personnelles. Le document brut reste dans le stockage local jusqu’à suppression ; le masquage ne rend donc pas le système adapté à des données personnelles réelles.

Les logs structurés portent le chemin de route, statut, durée et identifiant de corrélation. Les événements d’audit n’enregistrent que des métadonnées sûres (par exemple taille, type, nombre de citations), pas le texte ni la question. Toute nouvelle journalisation doit respecter cette règle.

Le contenu importé est non fiable. Le cas adversarial immuable `v7-x02` a montré que le moteur figé
pouvait sélectionner et restituer une instruction injectée. Après cette évaluation, le runtime a
centralisé quatre couches distinctes : règles système, question utilisateur, contenu documentaire
non fiable et extrait admis comme preuve. Le même garde valide réponses, citations, évaluations de
candidats et extractions persistées ; il restaure aussi le contexte après fragmentation, normalise
ponctuation et quelques homoglyphes courants. Les tests couvrent notamment faux administrateur,
divulgation de prompt, exfiltration, faux résultat attendu, instruction dirigée vers l'assistant et
dissimulée comme obligation contractuelle, formulations visant `system`, `tool`, le modèle de
langage ou l'agent IA, français/anglais, multi-ligne et contradiction avec la question. Aucun score
v7 n'a été recalculé après ce correctif.

Ce contrôle déterministe réduit les contournements testés mais ne garantit pas une résistance
universelle aux prompt injections ou obfuscations. La démo reste strictement synthétique et le
projet ne doit pas être présenté comme sûr contre toutes les injections. Si un fournisseur LLM ou
des outils sont ajoutés, ils exigent une nouvelle analyse de menace et une validation indépendante.

## Secrets et réseau

`.env.example` et `compose.yaml` contiennent uniquement des valeurs locales de démonstration. Elles doivent être remplacées par des secrets gérés hors dépôt avant tout déploiement. Ne jamais publier un volume Docker local, un `.env` réel, une clé de fournisseur ou une sauvegarde PostgreSQL.

Les quatre ports publiés par Compose sont liés explicitement à `127.0.0.1`. Redis et PostgreSQL restent joignables par les services sur le réseau Compose, mais pas par une interface LAN via ces publications. L'overlay public active les quotas mono-processus, désactive les comptes administrateur/lecteur, exige deux secrets runtime et bloque `/metrics`. Il n'active pas TLS, SSO, quota distribué, sauvegardes chiffrées ni isolation réseau de production. Nginx applique les en-têtes documentés dans l’architecture, mais toute exposition Internet exige HTTPS, secrets externes, contrôles d’accès, sauvegardes et revue d’infrastructure.

Les actions GitHub et les images externes exécutables sont épinglées à un commit SHA-1 ou digest SHA-256. Chaque service qui consomme l’alias local API possède son propre `build` et `pull_policy: build`, y compris lors d’un lancement ciblé normal. `scripts/check_supply_chain_refs.py` bloque le retour de tags mutables ou d’un consommateur local sans reconstruction forcée. Un opérateur ayant accès au démon Docker peut encore contourner volontairement cette politique avec `--no-build` ou exécuter une autre image ; ce contrôle ne prétend pas résister à l’administrateur local. L’épinglage n’établit pas à lui seul la provenance, la signature ou l’absence de vulnérabilité de l’artefact ; les mises à jour restent manuelles et doivent être revérifiées.

## Vérifications intégrées

La CI exécute Ruff, mypy, pytest avec couverture, le contrôle des références immuables, `pip-audit`, lint/typecheck/tests/build Playwright, `npm audit`, un test de stack Docker et Gitleaks. Ces contrôles réduisent le risque ; ils ne constituent pas une certification ni une revue de sécurité exhaustive.

Le **Codex Security Deep Scan n'a pas été exécuté par décision utilisateur**. Aucun rapport,
couverture ou verdict de Deep Scan n'est revendiqué. La revue de publication reste limitée aux
contrôles locaux explicitement listés dans
[`docs/release-validation.md`](release-validation.md).

Le contrôle local final ajoute Trivy sur les images construites. Le runtime API est multi-stage : le
binaire `uv` utilisé pour installer les dépendances reste dans le builder et n'est pas copié dans
l'image. Les couches finales Debian et Alpine appliquent les mises à jour de sécurité disponibles au
moment du build. L'upgrade au build améliore la fraîcheur mais rend la couche système moins
reproductible qu'un snapshot de dépôt immuable.

L'image web déclare l'utilisateur `nginx` non-root et prépare son fichier PID avec des droits
bornés. Ce correctif a été vérifié dans la pile réelle et par le contrôle de configuration Trivy.

### Analyse Trivy du 28 août 2026

Le scan brut, sans `--ignore-unfixed`, contredit l'ancien claim de zéro vulnérabilité : l'image web
Alpine rapporte `0` critique/élevée, mais l'image API Debian 13.6 rapporte `16` occurrences, soit
`13` CVE uniques (`3` critiques et `10` élevées uniques). Aucun `FixedVersion` n'est proposé par la
base Trivy 0.71.2 du jour ; le même scan avec `--ignore-unfixed` rapporte donc `0` corrigeable. Ce
second chiffre ne doit jamais être présenté seul.

| Surface | CVE Trivy | Analyse de reachabilité actuelle |
|---|---|---|
| `perl-base` | CVE-2026-13221, -42496, -42497, -48962, -57432, -57433, -8376, -9538 | Perl est présent dans Debian, mais aucune route API/worker ne l'invoque, ne traite une archive via `Archive::Tar` ou ne lui fournit regex/glob/désérialisation. L'image est `amd64`, alors que -8376 vise le chemin 32 bits. Présence du binaire = surface résiduelle, pas preuve d'exploitabilité. |
| `libsqlite3-0` | CVE-2026-11822, -11824 | Python fournit `sqlite3`, mais le produit utilise PostgreSQL/asyncpg et aucune route ne charge une base SQLite ou des données FTS5. |
| ncurses/tinfo | CVE-2025-69720, quatre paquets | Aucun TUI, terminal ou terminfo contrôlé par un document n'est utilisé par l'application. |
| `gzip` | CVE-2026-41992 | Les imports acceptent PDF/TXT/Markdown, pas d'archive, et le code API/worker n'exécute pas `gzip`. |
| `libacl1` | CVE-2026-54369 | Aucun appel ACL/setfacl n'existe dans les chemins applicatifs ; le stockage utilise une racine et des clés UUID. |

Cette analyse réduit la reachabilité observée mais n'annule pas les advisories. Avant exposition
Internet, il faut reconstruire avec une base corrigée ou réduire la base runtime, rescanner sans
masquer les non corrigées et réévaluer les chemins. Le scan web reste à zéro critique/élevée. Les
rapports JSON locaux sont des artefacts de validation de session et ne sont pas une attestation
signée.

### Candidat Qwen retiré des dépendances par défaut

Qwen reste un candidat historique rejeté avec manifest et adaptateur reproductibles. Sa dépendance
`llama-cpp-python` et ses transitives `diskcache`/`jinja2` ont été retirées du groupe de développement
et de la CI de la release candidate : aucun advisory n'est désormais ignoré pour les conserver.
Relancer cette expérience exige une installation locale volontaire, la vérification du manifest et
un audit des dépendances au moment de l'essai ; elle ne fait pas partie du runtime livré.
