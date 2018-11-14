from ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# Add dependencies
RUN pip3 install numpy scipy pandas
RUN pip3 install matplotlib
# dev
RUN pip3 install flake8 jupyter
# docs
RUN pip3 install sphinx sphinx-rtd-theme
# cover
RUN pip3 install coverage codecov

# Make sure we haven't missed any dependencies
ADD . /app
WORKDIR /app
RUN python3 setup.py install
