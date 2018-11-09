from ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install numpy scipy matplotlib

RUN pip3 install pandas

RUN pip3 install sphinx sphinx_rtd_theme

WORKDIR /app
