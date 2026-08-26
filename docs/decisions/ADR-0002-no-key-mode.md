# ADR-0002 — Mode sans clé extractif par défaut

## Décision

Le mode par défaut reste `extractive-local`, sans appel à un fournisseur externe ni clé API. Son embedding est le modèle local ONNX `paraphrase-multilingual-minilm-l12-v2-onnx-q@faf4aa4225822f3bc6376869cb1164e8e3feedd0`; la recherche hybride combine dense et lexical. Le lexical reste une voie legacy et un composant du classement hybride, pas l'espace vectoriel.

## Raisons

Tests, CI et démonstration doivent fonctionner sans coût ni secret. Les réponses restent extractives, liées à des passages, et peuvent s'abstenir ; ce mode n'est pas présenté comme un LLM complet.

## Limites acceptées

Le modèle est local et CPU, mais son téléchargement initial et son cache doivent être disponibles ou vérifiés. L'espace vectoriel est identifié par `embedding_model_id`; les chunks issus du feature hashing historique (`deterministic-hash-v1:384`) ne sont pas interchangeables avec le modèle ONNX. Un fournisseur local ou externe ultérieur exige mode explicite, coût explicite et nouvelle revue sécurité/évaluation.
