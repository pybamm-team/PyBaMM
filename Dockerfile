from ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python-pip

WORKDIR /app
