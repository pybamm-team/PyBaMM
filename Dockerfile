from ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# Add dependencies
RUN pip3 install numpy scipy pandas
RUN pip3 install matplotlib

# Make sure we haven't missed any dependencies
ADD . /app
WORKDIR /app
RUN python3 setup.py install
