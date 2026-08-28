# Démonstration EvidenceDesk en moins de trois minutes

Prérequis : `docker compose up --build --wait --wait-timeout 600` terminé et interface ouverte sur
`http://localhost:8080`. Utiliser uniquement les fichiers synthétiques versionnés.

Commencer par lire le bandeau : **Research prototype using synthetic data only. Not validated for
production, legal, medical, financial or compliance decisions.** Ce parcours administrateur est la
preuve locale complète. L'overlay public n'active que `demo.analyst` et ne donne donc accès ni à
l'audit, ni à l'état administrateur, ni au lancement d'une évaluation.

## 0:00–0:25: connexion et dossier

1. Se connecter avec `demo.admin` et le mot de passe local du README.
2. Ouvrir **Northwind supplier review** et signaler le bandeau « documents synthétiques ».
3. Montrer les trois fichiers déjà terminés et leurs passages masqués.

## 0:25–0:55: traitement asynchrone

1. Cocher l'attestation synthétique.
2. Importer `examples/demo-supplier-note.md`, présent dans l'allowlist synthétique publique.
3. Montrer l'état **En attente** ou **Traitement**, puis **Terminé** après le polling.
4. Expliquer que la réponse HTTP initiale est `queued` et que Redis/ARQ exécute le traitement.

Le worker local est rapide : l'état intermédiaire peut ne rester visible qu'une seconde. Le test
Playwright vérifie la réponse HTTP `202 queued`, puis attend le statut terminal réel.

## 0:55–1:35: réponse sourcée et recherche locale

1. Poser : **Quel montant de plateforme annuel est indiqué ?**
2. Montrer `Annual platform fee: EUR 48,000`.
3. Activer la citation `northstar_master_services_agreement.pdf · page 1`.
4. Vérifier que le panneau source sélectionne le document et affiche exactement le passage.
5. Indiquer le mode affiché : recherche hybride locale avec embedding ONNX, sans clé ni appel
   fournisseur externe. Ce résultat est une démonstration de parcours, pas une validation de qualité.

## 1:35–1:55: abstention

1. Poser : **Quel est le numéro TVA de Northstar ?**
2. Montrer le statut **Abstention explicite**, sans citation ni numéro inventé.

## 1:55–2:30: extraction et PII

1. Ouvrir l'extraction du contrat Northstar.
2. Montrer organisation, date, montant, obligations et boutons de preuve.
3. Sélectionner `supplier_register.txt` et montrer `[EMAIL REDACTED]` et `[PHONE REDACTED]`.

## 2:30–3:00: audit et résultat v7

1. Montrer `document.upload`, `document.process`, `dossier.ask` et les UUID de corrélation.
2. Montrer le résultat holdout v7 : **FAIL**. Le seul raw enregistré sur 40 cas rapporte 80 % de
   précision de citation, 36 % de cas répondables corrects, 80 % d'abstention correcte, F1
   d'extraction 45,67 %, Recall@5 100 %, zéro erreur et 0 USD de coût externe.
3. Dire explicitement : le moteur local et le parcours fonctionnent, mais ce holdout indépendant ne
   valide pas la qualité RAG. Aucune seconde exécution ni v8 n'est créé dans ce cycle. Les préflights
   v4 et v5 n'ont produit aucune métrique et ne sont pas rejoués.
4. Signaler que l'injection découverte sur `v7-x02` a reçu un correctif et des tests adversariaux
   après l'évaluation, sans réexécuter v7 ni revendiquer une meilleure métrique.

Après la démo, l'administrateur peut supprimer le document importé ; le fichier, les chunks et
l'extraction sont retirés, tandis qu'un audit minimal demeure.
