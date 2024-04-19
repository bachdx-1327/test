docker build -t sd-poc .
docker run --gpus all --rm --name sd-poc-container sd-poc