# Démonstration EvidenceDesk — moins de trois minutes

Prérequis : `docker compose up --build --wait --wait-timeout 180` terminé et interface ouverte sur
`http://localhost:8080`. Utiliser uniquement les fichiers synthétiques versionnés.

## 0:00–0:25 — Connexion et dossier

1. Se connecter avec `demo.admin` et le mot de passe local du README.
2. Ouvrir **Northwind supplier review** et signaler le bandeau « documents synthétiques ».
3. Montrer les trois fichiers déjà terminés et leurs passages masqués.

## 0:25–0:55 — Traitement asynchrone

1. Cocher l'attestation synthétique.
2. Importer `examples/demo-supplier-note.md`, présent dans l'allowlist synthétique publique.
3. Montrer l'état **En attente** ou **Traitement**, puis **Terminé** après le polling.
4. Expliquer que la réponse HTTP initiale est `queued` et que Redis/ARQ exécute le traitement.

Le worker local est rapide : l'état intermédiaire peut ne rester visible qu'une seconde. Le test
Playwright vérifie la réponse HTTP `202 queued`, puis attend le statut terminal réel.

## 0:55–1:35 — Réponse et preuve

1. Poser : **Quel montant de plateforme annuel est indiqué ?**
2. Montrer `Annual platform fee: EUR 48,000`.
3. Activer la citation `northstar_master_services_agreement.pdf · page 1`.
4. Vérifier que le panneau source sélectionne le document et affiche exactement le passage.

## 1:35–1:55 — Abstention

1. Poser : **Quel est le numéro TVA de Northstar ?**
2. Montrer le statut **Abstention explicite**, sans citation ni numéro inventé.

## 1:55–2:30 — Extraction et PII

1. Ouvrir l'extraction du contrat Northstar.
2. Montrer organisation, date, montant, obligations et boutons de preuve.
3. Sélectionner `supplier_register.txt` et montrer `[EMAIL REDACTED]` et `[PHONE REDACTED]`.

## 2:30–3:00 — Audit et évaluation

1. Montrer `document.upload`, `document.process`, `dossier.ask` et les UUID de corrélation.
2. Montrer les cartes holdout en **FAIL** : le produit ne transforme pas un résultat négatif en
   chiffre marketing.
3. Conclure : pile locale fonctionnelle, qualité RAG encore insuffisante sur holdout.

Après la démo, l'administrateur peut supprimer le document importé ; le fichier, les chunks et
l'extraction sont retirés, tandis qu'un audit minimal demeure.
