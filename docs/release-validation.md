# Validation locale de la release candidate

Date : 27 août 2026. Commit de départ :
`7a836458984ce3f79ebb8e64912bb735682a7b6b`. Branche : `main`. Aucun remote Git configuré.

Environnement observé : Ubuntu 24.04.4 LTS sous WSL2, noyau
`6.18.33.2-microsoft-standard-WSL2`, x86_64, Intel Core i5-12600KF, 16 CPU logiques,
16 768 458 752 octets de RAM, Docker 29.7.2, Compose 5.5.0, uv 0.11.7, Python 3.12.13 dans
l'environnement du projet, Node 24.15.0.

## Qualité logicielle

| Contrôle | Commande | Résultat |
|---|---|---|
| Backend complet | `uv run pytest --cov --cov-report=term-missing --cov-report=json:artifacts/coverage.json` | 345 réussis en 39,58 s ; couverture totale 82,52 % |
| Types Python | `uv run mypy` | 84 sources contrôlées, zéro erreur |
| Lint Python | `uv run ruff check apps/api apps/worker evals scripts` | réussi |
| Frontend | `npm run lint && npm run typecheck && npm run test -- --run && npm run build` | 6 tests réussis ; lint, types et build réussis |
| Navigateur réel | `E2E_REAL_API=1 ... npx playwright test e2e/main.spec.ts --workers=1` | 1 parcours métier réel réussi, 1 test mock volontairement ignoré ; axe-core sans violation critique/sérieuse |
| Capture séquentielle | `npx playwright test e2e/audit-capture.spec.ts --workers=1` | 1 réussi ; quatre captures recréées depuis la pile réelle |
| Supply chain | `uv run python scripts/check_supply_chain_refs.py` | références exécutables immuables confirmées |
| Dépendances Python | `uv run pip-audit --strict` | aucune vulnérabilité connue |
| Dépendances frontend | `npm audit --omit=dev --audit-level=high` | zéro vulnérabilité |
| Secrets | `gitleaks dir . --redact --max-target-megabytes 20` | zéro finding ; gros binaires modèle ignorés |

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

Le scan Codex Security initial a trouvé deux findings faibles : quota de requêtes absent et
instruction documentaire restituable. Les deux ont été corrigés. Une revue adversariale indépendante
a tenté des injections mono/multilignes, françaises/anglaises, ponctuées et Unicode à travers les
réponses legacy/v3, Qwen, NLI, citations, évaluations candidates et extractions persistées ; les
canaux vérifiés refusent désormais ces valeurs. Une seconde revue a trouvé puis fait corriger une
instruction adressée à l'assistant et déguisée en obligation contractuelle ; les contre-exemples
d'obligations opérationnelles normales restent admis. La revue finale a aussi vérifié les variantes
adressées à `system`, `tool`, au modèle de langage et à l'agent IA : 155 tests ciblés ont réussi sur
les voies déterministe, extraction et Qwen. La détection reste déterministe et bornée, sans claim
d'immunité universelle.

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
