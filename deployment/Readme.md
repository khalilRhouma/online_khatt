## Run the app locally:

Make sure that you have checkpoint under `models/`

```
export FLASK_APPLICATION_SETTINGS=config/app-settings.cfg
export FLASK_ENV=development
export FLASK_DEBUG=0
export FLASK_APP=main
python -m flask run --host=0.0.0.0
```

## Run the app with docker

Make sure that you have checkpoint under `models/`

Build docker image:

```
docker build  -f deployment/Dockerfile.deployment . -t deployment
```

Run image:

```
docker run --rm -it  -p 5000:5000/tcp deployment
```
