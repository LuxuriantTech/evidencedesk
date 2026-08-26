# ADR-0002 — Mode sans clé extractif par défaut

## Décision

Le mode par défaut est `extractive-local` avec feature hashing déterministe et recherche PostgreSQL hybride. Il n’appelle aucun fournisseur externe.

## Raisons

Les tests, la CI, la démonstration et les évaluations doivent être exécutables sans coût ni secret. Les réponses peuvent donc rester liées à un passage exact et s’abstenir sans génération libre.

## Limites acceptées

Le vecteur déterministe n’est pas un embedding sémantique pré-entraîné ; il est surtout lexical. Le
système n’est pas présenté comme un LLM complet. `ProviderBundle` expose les protocoles
`EmbeddingProvider` et `AnswerProvider` ; le registre n'accepte actuellement que
`extractive-local`. Un mode local ou externe ultérieur exige une implémentation explicite, un accord
sur les coûts et une nouvelle revue sécurité/évaluation.
