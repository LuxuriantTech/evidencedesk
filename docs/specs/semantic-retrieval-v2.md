# Spécification — Recherche sémantique et extraction v2

**Author:** EvidenceDesk
**Date:** 2026-08-26
**Status:** Approved
**Approbation :** critères imposés par Ardian Mehaj le 2026-08-26
**Baseline immuable :** `20380acdcd8861f5b57bb799efe490f65de87f4a`

## Contexte

La baseline EvidenceDesk offre un parcours de bout en bout, mais son encodage par
feature hashing n'est pas un modèle sémantique appris. Les holdouts v2 et v3 ont
également révélé des échecs de récupération/citation, d'abstention et
d'extraction. Leurs contenus et résultats sont placés en quarantaine : ils ne
seront consultés qu'après le gel du moteur et ne serviront ni à choisir une
méthode, ni à ajuster un seuil.

Cette version doit comparer des méthodes locales sans API payante exclusivement
sur un nouveau jeu de développement versionné. Le moteur retenu et sa
configuration seront ensuite gelés avant la création indépendante d'un holdout
v4 inédit, exécuté exactement une fois.

Le candidat dense pré-enregistré est
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, distribué sous
Apache-2.0, 384 dimensions, environ 118 millions de paramètres et 50 langues.
L'inférence CPU utilisera l'export ONNX Qdrant au commit exact
`faf4aa4225822f3bc6376869cb1164e8e3feedd0`, via `fastembed==0.8.0`.

## Functional Requirements

- FR-1: Le système MUST fournir un encodeur ONNX local, multilingue, de 384 dimensions et sans clé externe.
- FR-2: Le feature hashing MUST rester disponible uniquement comme baseline explicite de comparaison et comme double de test ; le runtime MUST NOT y revenir silencieusement.
- FR-3: L'encodeur MUST accepter une chaîne ou un lot, normaliser les vecteurs et exposer un identifiant de modèle immuable.
- FR-4: Chaque chunk MUST stocker son `embedding_model_id`, et une recherche dense MUST ignorer les vecteurs produits par un autre espace d'embedding.
- FR-5: Le banc de développement MUST comparer `lexical`, `dense`, `hybrid` et `hybrid_rerank`; l'hybride MUST employer une fusion de rangs qui ne suppose pas des scores bruts calibrés.
- FR-6: Le reranker MAY être retenu seulement s'il corrige au moins un couple réponse/citation sur le développement sans régression sur les autres métriques obligatoires ; une égalité MUST privilégier la méthode la plus simple et la plus rapide.
- FR-7: Une citation correcte MUST faire correspondre exactement l'identifiant du document et la page attendus, et son extrait normalisé MUST être contenu dans la page source attendue.
- FR-8: Une réponse MUST exiger une preuve de l'intention et de la valeur demandées ; la similarité de noms ou d'entités seule MUST NOT suffire.
- FR-9: Un cas `unanswerable` ou `adversarial` MUST produire `abstained`; un cas `ambiguous` MUST produire `ambiguous`.
- FR-10: L'extraction `supplier-v1` MUST couvrir organisation, type de document, date d'effet, date de renouvellement, montants, obligations, responsables et risques à partir de libellés normalisés, descriptions de champs et types génériques, sans constante issue des holdouts v2/v3.
- FR-11: Chaque valeur extraite MUST avoir une citation stricte ; les comparaisons MUST utiliser une normalisation typée (date, durée, montant ou texte), jamais une simple inclusion bidirectionnelle globale.
- FR-12: Un nouveau corpus de développement synthétique MUST être versionné et commité avant le premier benchmark ; aucun réglage MUST utiliser v2, v3 ou v4.
- FR-13: Chaque benchmark de développement MUST enregistrer méthodes, compteurs bruts, métriques, paramètres, graines, empreintes des entrées, révisions de modèle, temps, matériel et coût externe.
- FR-14: Le code et la configuration retenus MUST être committés avant la génération du holdout v4.
- FR-15: Le holdout v4 MUST être créé par un agent indépendant après le gel, comporter exactement 40 cas dont au moins 25 répondables, 10 sans réponse et 5 ambigus ou adversariaux, plus des attentes d'extraction sourcées ; son contenu MUST rester aveugle au développeur avant l'exécution.
- FR-16: L'exécution v4 MUST prendre un verrou avant tout calcul, refuser une sortie ou un verrou existants et enregistrer commits moteur/dataset, empreintes, modèle et configuration.
- FR-17: Après l'ouverture de v4, le code, la configuration, le protocole et le gold MUST NOT changer ; les recalculs MUST utiliser uniquement l'artefact brut.
- FR-18: Les seuils de succès v4 MUST rester : précision des citations >= 0,90, exactitude d'abstention >= 0,85, F1 d'extraction >= 0,90 et taux d'erreur = 0.
- FR-19: Les protections existantes d'authentification, RBAC, PII, audit, idempotence, reprise, suppression et validation de fichiers MUST rester couvertes par leurs tests.
- FR-20: Le travail MUST rester local, sans publication, déploiement, fournisseur payant ni modification de profil ou de CV.

## Non-Functional Requirements

### Performance et reproductibilité

- NFR-P1 : Sur le matériel de référence CPU, une recherche chaude SHOULD avoir une latence médiane < 100 ms et p95 < 250 ms, hors parsing et démarrage du modèle.
- NFR-P2 : La réponse extractive complète SHOULD avoir une latence p95 < 500 ms sur le corpus de démonstration.
- NFR-P3 : Le téléchargement du modèle SHOULD rester < 350 MB et le RSS du processus SHOULD rester < 1,5 GiB pendant le benchmark.
- NFR-P4 : L'indexation MUST encoder les chunks par lots, avec ordre de sortie déterministe.
- NFR-R1 : Les versions Python, paquet, modèle et fichiers ONNX MUST être figées ; les fichiers modèle MUST être vérifiés par SHA-256.
- NFR-R2 : Le runtime MUST utiliser les seuls fichiers locaux et échouer explicitement si le modèle ou son empreinte manque.

### Sécurité et qualité

- NFR-S1 : Aucun document complet, secret, token, mot de passe ou PII non masquée MUST apparaître dans les logs ou artefacts d'évaluation.
- NFR-S2 : Les audits de secrets et de dépendances MUST rester sans vulnérabilité critique connue ; tout finding important restant MUST être documenté.
- NFR-Q1 : La couverture combinée MUST rester >= 80 % ; les modules métier modifiés SHOULD atteindre 80 % de couverture statements.
- NFR-A1 : Aucun changement de parcours UI n'est requis ; les tests d'accessibilité du parcours principal MUST rester sans violation sérieuse ou critique.

## Acceptance Criteria

### AC-1: Modèle local figé (FR-1, FR-2, FR-3, NFR-R1, NFR-R2)

Given une machine sans clé ni réseau au runtime,
When EvidenceDesk charge le modèle local vérifié et encode un lot,
Then il produit des vecteurs normalisés de 384 dimensions avec l'identifiant pré-enregistré,
et échoue clairement si un fichier attendu est absent ou altéré.

### AC-2: Isolation des espaces vectoriels (FR-4)

Given des chunks encodés par deux modèles,
When une recherche dense est lancée avec le modèle retenu,
Then seuls les chunks portant exactement son identifiant sont candidats.

### AC-3: Comparaison reproductible (FR-5, FR-6, FR-12, FR-13)

Given le jeu de développement commité,
When le benchmark compare les quatre méthodes avec la même graine,
Then un artefact contient les résultats bruts, métriques, empreintes, paramètres, latences et matériel,
et la règle de sélection désigne une configuration unique sans lire v2/v3.

### AC-4: Citations strictes (FR-7, FR-11)

Given une réponse avec le bon texte mais la mauvaise page ou le mauvais document,
When l'évaluateur strict la note,
Then la citation est incorrecte ; seul un extrait normalisé présent dans la bonne page est accepté.

### AC-5: Abstention stricte (FR-8, FR-9)

Given une question sans preuve, contradictoire ou ambiguë,
When le moteur répond,
Then son statut respecte exactement le type de cas et aucune réponse certaine n'est émise.

### AC-6: Extraction généralisée (FR-10, FR-11)

Given des libellés et formulations synthétiques inédits du jeu de développement,
When l'extracteur produit `supplier-v1`,
Then les huit catégories sont typées, sourcées et notées par normalisation de domaine.

### AC-7: Gel avant v4 (FR-14, FR-15)

Given une configuration choisie sur développement,
When le holdout v4 indépendant est généré,
Then le commit moteur le précède et l'attestation v4 confirme sa composition et son absence de chevauchement.

### AC-8: Exécution v4 unique (FR-16, FR-17)

Given le holdout v4 commité et jamais exécuté,
When la commande one-shot démarre,
Then elle crée d'abord un verrou exclusif, refuse toute seconde exécution et produit un artefact brut lié aux deux commits.

### AC-9: Recalcul indépendant (FR-17, FR-18)

Given l'artefact brut v4 en lecture seule,
When un second évaluateur recalcule les métriques sans relancer le moteur,
Then les compteurs et métriques correspondent exactement et le verdict applique les seuils pré-enregistrés.

### AC-10: Non-régression (FR-19, NFR-Q1, NFR-A1)

Given la version gelée,
When les suites API, worker, RBAC, idempotence, reprise, citations, abstention, PII, audit, frontend, E2E et accessibilité sont exécutées,
Then elles restent vertes et la couverture combinée est au moins 80 %.

### AC-11: Coût et matériel honnêtes (FR-1, FR-13, NFR-P1, NFR-P2, NFR-P3)

Given le benchmark local,
When son rapport est produit,
Then il indique CPU, RAM, modèle, taille disque, RSS, temps médian/p95 et coût API externe de 0 EUR sans présenter le mode extractif comme un LLM.

### AC-12: Limites d'autorisation (FR-20)

Given l'absence de GO public,
When la livraison locale est terminée,
Then aucun push, dépôt public, déploiement, CV ou profil n'a été modifié.

## Edge Cases

- EC-1: modèle absent, incomplet ou hash invalide -> arrêt explicite avant indexation.
- EC-2: texte vide ou uniquement masqué -> vecteur nul interdit, document marqué en échec contrôlé.
- EC-3: PostgreSQL/Redis indisponible -> état de tâche en échec/reprise selon la politique existante, sans fallback non audité.
- EC-4: deux workers indexent le même document -> l'idempotence existante empêche les chunks dupliqués.
- EC-5: aucun score ne franchit les preuves d'intention et de valeur -> abstention.
- EC-6: deux passages plausibles se contredisent -> statut ambigu et citations des passages conflictuels.
- EC-7: bonne valeur sur mauvaise page -> citation et champ d'extraction notés faux.
- EC-8: sortie/lock v4 déjà présent -> refus avant chargement du corpus et avant inférence.
- EC-9: reranker indisponible -> méthode non sélectionnée ; aucun fallback silencieux pendant une mesure.

## API Contracts

### POST /api/v1/documents/{document_id}/questions

L'endpoint conserve le contrat existant. La réponse inclut `status`, `answer`,
`mode` et des citations `{document_id, page_number, excerpt}`. Les stratégies
et détails de score restent internes afin de ne pas exposer un contrat instable.

### Contrats internes

```python
class EmbeddingProvider(Protocol):
    model_id: str
    dimensions: int
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

class RetrievalMethod(str, Enum):
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"

class EvaluationResultV2(TypedDict):
    engine_commit: str
    dataset_commit: str
    dataset_sha256: str
    config_sha256: str
    model_id: str
    model_revision: str
    seed: int
    raw_cases: list[dict[str, object]]
    metrics: dict[str, float]
    timings_ms: dict[str, float]
    hardware: dict[str, object]
```

L'API REST publique existante garde ses formes de requête/réponse. Le champ
`mode` MUST identifier honnêtement `extractive-local-onnx`; aucune migration de
contrat frontend n'est nécessaire.

## Data Models

### Chunk

| Champ | Type | Contraintes |
|---|---|---|
| `embedding` | `vector(384)` | nullable pendant migration/indexation |
| `embedding_model_id` | `varchar(160)` | non nul pour tout vecteur, filtré à la recherche |
| `page_number` | entier | >= 1 |
| `section` | texte | libellé source normalisé |
| `content` | texte | contenu masqué, jamais loggé |

### Manifeste du modèle

| Champ | Type | Contraintes |
|---|---|---|
| `model_id` | texte | immuable |
| `source` | URL | dépôt officiel |
| `revision` | SHA git | exact |
| `dimensions` | entier | 384 |
| `license` | texte | Apache-2.0 |
| `files` | liste | chemin, SHA-256 et taille mesurée |

## Out of Scope

- OS-1: fournisseur LLM payant, fine-tuning ou génération libre.
- OS-2: OCR, documents réels ou données personnelles réelles.
- OS-3: changement visuel important de l'interface.
- OS-4: nouvelle optimisation après ouverture du holdout v4.
- OS-5: publication, push, déploiement ou modification de carrière.
- OS-6: garantie WORM externe du verrou one-shot ; le contrôle est local, versionné et vérifiable.
- OS-7: GPU obligatoire ; il peut accélérer des essais, mais la référence reste CPU.

## Questions ouvertes

Aucune. La méthode finale et les seuils numériques seront déterminés uniquement
par le benchmark de développement conformément à FR-6 et seront consignés dans
un fichier de configuration gelé avant v4.
