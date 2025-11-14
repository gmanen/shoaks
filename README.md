# Shoaks Docker workflow

Ce dépôt inclut désormais une configuration Docker pour développer et tester l'application Vite.

## Construction de l'image

La construction utilise l'image de base `node:20-alpine` et exécute `npm ci` (ou `npm install` si aucun `package-lock.json` n'est présent). Avant de lancer la construction pour la première fois, générez un `package-lock.json` local si nécessaire :

```bash
npm install --package-lock-only
```

Puis construisez l'image :

```bash
docker compose build
```

## Développement avec hot-reload

Lancez le serveur de développement Vite via Docker avec le hot-reload (grâce à `stdin_open: true` et `tty: true`) :

```bash
docker compose up
```

Le serveur est exposé sur [http://localhost:5173](http://localhost:5173).

## Lancer les tests

Vous pouvez exécuter les tests Mocha directement dans le même conteneur :

```bash
docker compose run app npm test
```

Un service dédié, désactivé par défaut, est également fourni. Activez le profil `tests` pour lancer les tests ou d'autres commandes (benchmarks par exemple) sans affecter le service de développement :

```bash
docker compose --profile tests run --rm tests
```

Adaptez la commande `command:` dans `docker-compose.yml` pour lancer `npm run benchmark:shoaks`, `npm run benchmark:layers`, etc.

## Arrêt et nettoyage

Arrêtez les services avec `Ctrl+C`, puis supprimez les conteneurs arrêtés si besoin :

```bash
docker compose down
```

Le volume nommé `node_modules` conserve les dépendances installées dans le conteneur afin d'accélérer les reconstructions.
