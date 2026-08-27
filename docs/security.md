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

Les imports sont limités à PDF/TXT/Markdown et à 10 MiB de fichier. Avant que Starlette analyse ou spoule un multipart, un middleware ASGI coupe toute requête dépassant 11 MiB, avec ou sans `Content-Length` ; Nginx applique la même enveloppe. Cela borne une requête individuelle, pas le volume agrégé de nombreuses connexions simultanées : aucune limitation de débit distribuée n’est implémentée. L’application valide ensuite type annoncé, signature et encodage. Le worker refuse par défaut plus de 200 pages, 1 000 000 de caractères extraits ou 5 000 chunks ; ces budgets sont configurables par `MAX_DOCUMENT_PAGES`, `MAX_EXTRACTED_CHARS` et `MAX_DOCUMENT_CHUNKS`. Les clés de stockage sont générées avec UUID ; une traversée de répertoire est refusée par résolution sous la racine locale.

La suppression via `DELETE /api/v1/documents/{document_id}` verrouille la tâche puis le document dans le même ordre que le worker, supprime le fichier local et les données dérivées, puis marque le document supprimé. Le worker revérifie ce marqueur sous verrou avant toute persistance et ses chemins d’échec ne peuvent pas le réactiver. Les métadonnées d’audit restent.

Le parsing PDF s’exécute dans un sous-processus `spawn` : le parent impose un délai mural et le child applique sous POSIX une limite d’espace d’adressage et de temps CPU. Les codes `pdf_processing_timeout` et `pdf_resource_limit` sont terminaux. La limite mémoire dure repose sur `resource` et n’est donc pas disponible sous Windows natif ; le parcours documenté et testé utilise Linux/WSL ou le conteneur Linux.

Limite résiduelle : la suppression du fichier et le commit PostgreSQL ne sont pas atomiques ; une panne DB entre les deux demanderait une réconciliation. La conservation est manuelle ; aucune purge planifiée, rétention chiffrée ou suppression cryptographique n’est implémentée. Pour toute donnée non synthétique, il faut définir une durée, des sauvegardes, une rétention des journaux et une purge vérifiable avant usage.

## PII, logs et prompt injection

Le worker remplace dans les chunks les e-mails, numéros de téléphone et identifiants synthétiques `SYN-ID-*`. C’est une démonstration de masquage par motifs, pas une détection complète de données personnelles. Le document brut reste dans le stockage local jusqu’à suppression ; le masquage ne rend donc pas le système adapté à des données personnelles réelles.

Les logs structurés portent le chemin de route, statut, durée et identifiant de corrélation. Les événements d’audit n’enregistrent que des métadonnées sûres (par exemple taille, type, nombre de citations), pas le texte ni la question. Toute nouvelle journalisation doit respecter cette règle.

Le contenu importé est non fiable. Le pipeline ne lui donne aucune capacité d’exécuter des instructions, d’appeler des outils ou de modifier le système. Le mode extractif ne suit pas d’instructions de document : il cite des passages. Si un fournisseur LLM est ajouté, la séparation stricte entre instructions système et documents, la validation des sorties, les limites d’outils et des tests d’injection deviennent obligatoires.

## Secrets et réseau

`.env.example` et `compose.yaml` contiennent uniquement des valeurs locales de démonstration. Elles doivent être remplacées par des secrets gérés hors dépôt avant tout déploiement. Ne jamais publier un volume Docker local, un `.env` réel, une clé de fournisseur ou une sauvegarde PostgreSQL.

Les quatre ports publiés par Compose sont liés explicitement à `127.0.0.1`. Redis et PostgreSQL restent joignables par les services sur le réseau Compose, mais pas par une interface LAN via ces publications. Le compose n’active pas TLS, SSO, limitation de débit, sauvegardes chiffrées ni isolation réseau de production. Nginx applique les en-têtes documentés dans l’architecture, mais toute exposition Internet exige HTTPS, secrets externes, contrôles d’accès, sauvegardes et revue d’infrastructure.

Les actions GitHub et les images externes exécutables sont épinglées à un commit SHA-1 ou digest SHA-256. Chaque service qui consomme l’alias local API possède son propre `build` et `pull_policy: build`, y compris lors d’un lancement ciblé normal. `scripts/check_supply_chain_refs.py` bloque le retour de tags mutables ou d’un consommateur local sans reconstruction forcée. Un opérateur ayant accès au démon Docker peut encore contourner volontairement cette politique avec `--no-build` ou exécuter une autre image ; ce contrôle ne prétend pas résister à l’administrateur local. L’épinglage n’établit pas à lui seul la provenance, la signature ou l’absence de vulnérabilité de l’artefact ; les mises à jour restent manuelles et doivent être revérifiées.

## Vérifications intégrées

La CI exécute Ruff, mypy, pytest avec couverture, le contrôle des références immuables, `pip-audit`, lint/typecheck/tests/build Playwright, `npm audit`, un test de stack Docker et Gitleaks. Ces contrôles réduisent le risque ; ils ne constituent pas une certification ni une revue de sécurité exhaustive.

### Advisory accepté sur un candidat expérimental

L'audit du 27 août 2026 signale `PYSEC-2026-2447` / `GHSA-w8v5-vhqr-4h9v` dans
`diskcache==5.6.3`, dépendance transitive de `llama-cpp-python==0.3.35`. L'advisory GitHub est de
sévérité modérée (CVSS 5,2), n'annonce aucune version corrigée et exige qu'un attaquant local puisse
écrire dans le répertoire de cache avant qu'une application lise la valeur pickle :
https://github.com/advisories/GHSA-w8v5-vhqr-4h9v.

EvidenceDesk n'instancie jamais `LlamaDiskCache`, n'appelle pas `set_cache` et n'utilise Qwen que
comme candidat de développement rejeté ; le mode retenu `grounded-local-v3`, l'API, le worker et la
démo n'exécutent pas `llama-cpp-python`. Le risque résiduel est donc limité à une réexécution locale
explicite de l'expérience Qwen par une personne ayant déjà accès au poste. L'exception est nommée
dans la commande CI au lieu de masquer toutes les vulnérabilités. Elle doit être supprimée dès
qu'une version corrigée de `diskcache` ou une dépendance `llama-cpp-python` sans ce paquet est
disponible. Aucun advisory critique ou élevé n'est ignoré.
