# Base Image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Install the necessary dependencies
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y libgmp3-dev libmpfr-dev libmpc-dev \
    && apt-get install -y libffi-dev libssl-dev

# Copy necessary files into the container
COPY setup.py  .
COPY CMakeBuild.py .
COPY README.md .
COPY pybamm/version.py ./pybamm/version.py

# Install PyBaMM
RUN pip install -e ".[dev]"


# Expose the default Jupyter notebook port
EXPOSE 8888

# Start Jupyter notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
