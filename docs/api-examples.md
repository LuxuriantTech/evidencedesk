# Exemples API reproductibles

Les exemples ci-dessous visent un compose local démarré par `docker compose up --build --wait`. Ils n’incluent aucun jeton, mot de passe ou secret réel. Définissez vos propres variables dans votre shell local.

```bash
export EVIDENCEDESK_BASE_URL=http://127.0.0.1:8080
export EVIDENCEDESK_USERNAME=demo.admin
export EVIDENCEDESK_PASSWORD='EvidenceDemo-Admin-2026!'
```

Obtenir un jeton :

```bash
TOKEN=$(curl --fail --silent \
  -X POST "$EVIDENCEDESK_BASE_URL/api/v1/auth/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode "username=$EVIDENCEDESK_USERNAME" \
  --data-urlencode "password=$EVIDENCEDESK_PASSWORD" \
  | jq -r '.access_token')
```

Lister les dossiers et vérifier la disponibilité :

```bash
curl --fail "$EVIDENCEDESK_BASE_URL/ready"
curl --fail -H "Authorization: Bearer $TOKEN" "$EVIDENCEDESK_BASE_URL/api/v1/dossiers"
```

Importer exclusivement un fichier synthétique dans un dossier choisi :

```bash
curl --fail -X POST "$EVIDENCEDESK_BASE_URL/api/v1/dossiers/<DOSSIER_UUID>/documents" \
  -H "Authorization: Bearer $TOKEN" \
  -F 'is_synthetic=true' \
  -F 'file=@examples/demo-supplier-note.md;type=text/markdown'
```

Poser une question sourcée et consulter l’extraction :

```bash
curl --fail -X POST "$EVIDENCEDESK_BASE_URL/api/v1/dossiers/<DOSSIER_UUID>/ask" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"question":"What is the annual service fee?"}'

curl --fail -H "Authorization: Bearer $TOKEN" \
  "$EVIDENCEDESK_BASE_URL/api/v1/dossiers/<DOSSIER_UUID>/extraction"
```

La réponse fournit `status`, `mode`, `correlation_id` et des citations avec document, page et extrait. Une réponse `abstained` signifie que le mode extractif n’a pas trouvé de preuve suffisamment fiable ; ne la transforme pas en affirmation.

Pour effacer un document synthétique importé pendant la démonstration :

```bash
curl --fail -X DELETE -H "Authorization: Bearer $TOKEN" \
  "$EVIDENCEDESK_BASE_URL/api/v1/documents/<DOCUMENT_UUID>"
```
