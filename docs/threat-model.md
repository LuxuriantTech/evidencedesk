# Modèle de menace simplifié

## Périmètre

EvidenceDesk est un prototype de recherche exécuté localement avec des données synthétiques. Le
périmètre couvre React/Nginx, FastAPI, PostgreSQL/pgvector, Redis/ARQ, le worker et le stockage local.
TLS, SSO, sauvegardes, stockage S3 et isolation multi-tenant ne sont pas implémentés.

## Actifs et frontières

| Actif | Frontière principale | Contrôle actuel |
|---|---|---|
| Fichiers synthétiques bruts | navigateur → API → volume | type, signature, taille, chemin UUID, allowlist publique |
| Chunks, embeddings, extractions | worker → PostgreSQL | budgets parseur, masquage PII, transaction et idempotence |
| Question et réponse | utilisateur/document → moteur | RBAC, séparation de confiance, preuve document/page/extrait |
| Mots de passe et JWT | navigateur → API | Argon2, JWT court, jeton seulement en mémoire |
| Audit et métriques | API/worker → stockage/observabilité | métadonnées limitées, corrélations, métriques non textuelles |

Le contenu documentaire est une entrée non fiable. Il reste visible dans le panneau document comme
donnée, mais ne doit pas devenir une instruction système, une réponse, une citation, une extraction
ou une évaluation candidate sans franchir la barrière d'admission.

## Menaces retenues

- instruction injectée dans un document et restituée comme réponse ;
- brute force ou épuisement CPU par répétition de login, question ou upload ;
- import d'un fichier réel ou non approuvé dans la démo publique ;
- traversée de chemin ou divergence MIME/signature ;
- PDF provoquant une expansion mémoire/CPU ;
- contournement RBAC par l'interface ;
- fuite de document, mot de passe, JWT ou PII dans les logs ;
- tâche rejouée, suppression concurrente ou état bloqué en traitement ;
- dépendance ou image mutable.

## Contrôles et limites

Le correctif post-v7 centralise la séparation système/question/document/preuve et assainit aussi les
extractions déjà persistées. Des tests multilingues couvrent plusieurs classes connues. Ce contrôle
est déterministe et borné : il ne prouve pas une résistance universelle aux prompt injections. Le
runtime n'appelle aucun outil et aucun LLM génératif, ce qui limite l'impact à l'intégrité de la
réponse et à l'ingénierie sociale.

La limitation de débit est une fenêtre glissante par adresse vue directement par l'API, en mémoire
du processus, avec `429` et `Retry-After`. L'application ne fait pas confiance à
`X-Forwarded-For`. Derrière le Nginx Docker fourni, tous les visiteurs partagent donc l'adresse du
proxy et le même bucket : ce garde-fou global peut limiter l'abus, mais un visiteur peut épuiser le
quota des autres. L'overlay est une préparation locale, pas une limitation fiable par visiteur. Une
exposition Internet exigerait un quota partagé au proxy ou dans Redis, avec une chaîne de proxies de
confiance explicitement configurée.

Le RBAC est global à l'espace de démonstration ; il n'existe aucune ACL par dossier ni isolation de
tenant. La démo publique désactive les comptes admin et lecteur et conserve un analyste limité, mais
un analyste voit l'ensemble du corpus synthétique. Ce modèle est incompatible avec des données de
plusieurs organisations.
