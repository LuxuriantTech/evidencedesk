# ADR-0005 — Recherche sémantique locale ONNX

## Décision

EvidenceDesk utilise `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q` au commit `faf4aa4225822f3bc6376869cb1164e8e3feedd0` comme embedding local CPU. Les vecteurs font 384 dimensions et sont étiquetés par `paraphrase-multilingual-minilm-l12-v2-onnx-q@<révision>`. Le classement retenu est hybride dense + lexical; le reranker de développement n'est pas retenu.

## Éléments factuels

Le manifest versionné déclare Apache-2.0, environ 118 M paramètres, maximum 512 tokens, pooling mean via `fastembed==0.8.0` et exécution `onnxruntime==1.29.0`/CPU. Le téléchargement mesuré est `266906689` octets, dont `235052644` pour l'ONNX.

## Conséquences et limites

Les embeddings feature-hashing historiques sont incompatibles avec cet espace vectoriel et doivent rester identifiables par leur `model_id`. La recherche lexicale est conservée pour compatibilité et fusion. Le benchmark de développement est in-memory; la recherche API SQL (20 candidats denses + 20 lexicaux) n'est pas validée à grande échelle. Le pooling mean est une convention runtime, non une preuve d'optimalité.
