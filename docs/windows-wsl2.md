# Windows et WSL2

La procédure effectivement vérifiée utilise Ubuntu sous WSL2 et des conteneurs Linux. Le lancement
Windows natif n'est pas revendiqué comme testé.

## Prérequis

- WSL2 avec Ubuntu ;
- Docker Desktop avec l'intégration WSL activée, ou Docker Engine joignable depuis WSL ;
- Git ;
- environ 4 Go d'espace libre pour images, volumes et modèle ONNX.

Dans un terminal Ubuntu WSL :

```bash
cd ~/dev/evidencedesk
docker version
docker compose version
docker compose up --build --wait --wait-timeout 600
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
```

Ouvrir ensuite `http://127.0.0.1:8080` depuis Windows. Les ports du Compose sont liés à loopback ;
ils ne constituent pas une publication Internet.

Pour arrêter sans effacer les données :

```bash
docker compose down
```

`docker compose down --volumes` reconstruit la base synthétique au prochain démarrage et supprime les
volumes EvidenceDesk locaux. Ne pas l'utiliser si ces volumes doivent être conservés.

## Limite Windows native

Le parseur PDF utilise `RLIMIT_AS` et `RLIMIT_CPU` sous POSIX. Ces limites fortes ne sont pas
disponibles dans un processus Python Windows natif ; le conteneur Linux ou WSL2 est donc le chemin
documenté. Aucun support GPU n'est nécessaire.
