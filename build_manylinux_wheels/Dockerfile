FROM quay.io/pypa/manylinux2010_x86_64

ENV PLAT manylinux2010_x86_64

RUN yum -y remove cmake
RUN yum -y install openblas-devel
RUN /opt/python/cp37-cp37m/bin/pip install --upgrade pip cmake
RUN ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake

COPY dependencies/ deps
COPY install_sundials.sh /install_sundials.sh
RUN chmod +x /install_sundials.sh
RUN ./install_sundials.sh