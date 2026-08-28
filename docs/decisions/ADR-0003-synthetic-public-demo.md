# ADR-0003: Démonstration publique synthétique uniquement

## Décision

Le corpus versionné et les imports de la démonstration publique sont synthétiques. `PUBLIC_DEMO_MODE` exige l’attestation `is_synthetic=true`.

## Raisons

Le stockage local, la rétention manuelle et le masquage PII par motifs ne sont pas suffisants pour recevoir des données personnelles, contrats ou dossiers réels sur une démo publique.

## Conséquences

L’attestation est un garde-fou d’interface/API, non une preuve matérielle de synthèse. L’import public de documents réels reste hors périmètre jusqu’à la mise en place et la vérification de contrôles de données adaptés.
