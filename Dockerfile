FROM python:3.11-slim

# Set the working directory
WORKDIR /PyBaMM

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential

# Copy project files into the container
COPY . .

# Install PyBaMM
RUN python -m pip install --upgrade pip
RUN pip install -e ".[all]"

CMD ["/bin/bash"]