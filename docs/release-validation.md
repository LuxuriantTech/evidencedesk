# Validation locale de la release candidate

Date : 28 août 2026. Commit de départ :
`a8e3d669d1494524765d3ec08aac0aaf848d8af9`. Branche : `main`. Aucun remote Git n'était
configuré au début de la validation.

Environnement observé : Ubuntu 24.04.4 LTS sous WSL2, noyau
`6.18.33.2-microsoft-standard-WSL2`, x86_64, Intel Core i5-12600KF, 16 CPU logiques,
16 768 458 752 octets de RAM, Docker 29.7.2, Compose 5.5.0, uv 0.11.7, Python 3.12.13 dans
l'environnement du projet, Node 24.15.0.

## Qualité logicielle

| Contrôle | Commande | Résultat |
|---|---|---|
| Backend complet | `uv run pytest --cov --cov-report=term-missing --cov-report=json:artifacts/coverage.json` | 347 réussis en 34,88 s ; couverture totale 82,49 % |
| Types Python | `uv run mypy` | 84 sources contrôlées, zéro erreur |
| Lint Python | `uv run ruff check apps/api apps/worker evals scripts` | réussi |
| Frontend | `npm run lint && npm run typecheck && npm run test -- --run && npm run build` | 6 tests réussis ; lint, types et build réussis |
| Navigateur réel | `E2E_REAL_API=1 ... npx playwright test e2e/main.spec.ts --workers=1` | 1 parcours métier réel réussi, 1 test mock volontairement ignoré ; axe-core sans violation critique/sérieuse |
| Capture séquentielle | `npx playwright test e2e/audit-capture.spec.ts --workers=1` | 1 réussi ; quatre captures recréées depuis la pile réelle |
| Supply chain | `uv run python scripts/check_supply_chain_refs.py` | références exécutables immuables confirmées |
| Dépendances Python | `uv run pip-audit --strict` | aucune vulnérabilité connue |
| Dépendances frontend | `npm audit --omit=dev --audit-level=high` | zéro vulnérabilité |
| Secrets | `gitleaks git --log-opts='--all' .` et `gitleaks dir .` | zéro finding sur tout l'historique et le worktree ; les dépendances locales ignorées par Git ne sont pas publiées |

Le Mypy configuré exclut uniquement les générateurs historiques immuables de holdouts v2 à v7. Il
couvre le runtime, l'évaluateur, les tests et les scripts maintenus. Aucun holdout n'a été lancé.

## Docker et démonstration

La pile locale a été reconstruite après suppression des seuls volumes synthétiques EvidenceDesk.
Tous les services `postgres`, `redis`, `api`, `worker` et `web` sont sains ; `/health` renvoie
`ok` et `/ready` renvoie `ready`. Le parcours Playwright vérifie connexion, import `202 queued`,
traitement worker terminal, réponse avec preuve, abstention, extraction, PII masquée, audit, résultat
v7 négatif et suppression contrôlée.

L'overlay public a été construit et démarré séparément avec des secrets temporaires non versionnés.
Vérifications HTTP observées : analyste connecté ; admin `401` ; reader `401` ; vue statut admin
`403` ; fichier inconnu `422 public_demo_file_not_approved` ; `/metrics` `404` ; sixième requête
d'authentification dans la fenêtre `429` ; `/health` et `/ready` réussis. Les ports sont restés sur
`127.0.0.1` et aucune publication Internet n'a été effectuée.

## Sécurité

Le **Codex Security Deep Scan n'a pas été exécuté par décision utilisateur**. Aucun rapport,
couverture ou verdict de Deep Scan n'est revendiqué. Cette validation repose sur une revue locale
limitée, des tests, Gitleaks, Trivy et les audits de dépendances ; elle ne leur attribue pas une
couverture équivalente.

Deux défauts faibles documentés auparavant — quota absent sur les routes sensibles et instruction
documentaire restituable — ont été corrigés et gardent des tests de non-régression. Les tests
adversariaux couvrent les variantes mono/multilignes, françaises/anglaises, ponctuées et Unicode sur
les réponses, citations, évaluations candidates et extractions persistées. La détection reste
déterministe et bornée, sans garantie d'immunité universelle.

Trivy configuration a aussi détecté que l'image web s'exécutait en root. Le runtime utilise
désormais l'utilisateur `nginx` et le nouveau test de dépôt empêche le retour d'un `USER` root ;
la pile réelle et le contrôle Trivy configuration passent après ce correctif.

Trivy brut rapporte zéro critique/élevée sur l'image web. L'image API Debian rapporte 16 occurrences,
13 CVE système uniques critiques/élevées, sans version corrigée dans la base du jour ; le scan des
seules vulnérabilités corrigeables rapporte zéro. Leur reachabilité et la dette de base image sont
analysées dans `docs/security.md`.

## Captures authentiques

| Fichier | SHA-256 |
|---|---|
| `docs/screenshots/01-dashboard-desktop.png` | `6e97a78a9c00d246c0833cd5e7fd55699db7dea4d2e2df1139f6ca36f58516fd` |
| `docs/screenshots/02-sourced-answer-desktop.png` | `96fe1c154a90b13f8e63e2eb07ab725c5fdcc2faffc2f90c63510a17f649876e` |
| `docs/screenshots/03-evaluation-results.png` | `af20c95e26265580f4d8c1fce4479eb6b95b275d3f72d084dcb18e82f65ed096` |
| `docs/screenshots/04-dashboard-mobile.png` | `fc44631165b1ed616f216fb1c172d05dd8554fa6c53fd4d6b79f4c5ae16cb3c4` |

## Limite empirique immuable

Le holdout v7 reste `FAIL` : Recall@5 100 %, exactitude répondable 36 %, précision/rappel des
citations 80 %/48 %, abstention 80 % et F1 extraction 45,67 %. Aucun correctif postérieur n'a été
utilisé pour le rejouer, le rescorrer ou remplacer ses métriques.
