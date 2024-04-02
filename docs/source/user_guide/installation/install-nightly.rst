Install from nightly releases
=============================

.. contents::

This page describes how to install nightly wheels for PyBaMM for Linux, macOS, and Windows. These wheels are
built every fortnight, and therefore are more like "fortnightlies" rather than "nightlies". This provides
a balance between getting to access the latest features and bug fixes, while also ensuring that the code
is *relatively* stable. However, these wheels are intended for users who need to access the latest features
and are willing to accept risks that come with unprecented bugs.

Installing the nightly wheels is similar to installing the stable wheels, with the exception that they are hosted
on a different PyPI-like index.

Prerequisites
-------------

First, ensure that you have Python installed on your system. You can download a Python interpreter from
the `official website <https://www.python.org/downloads/>`_ or from your package manager of choice (e.g. ``apt-get``,
``brew``, ``conda``, and so on).

Install command
---------------

.. danger::

    Please be careful and prefer to use the ``--index-url`` flag instead of ``--extra-index-url``, since through the
    latter it is possible to execute a "dependency confusion" supply chain exploit by maliciously uploading a package
    with the same name as that of a popular package to the PyPI index. This is a known security vulnerability, and
    the Python Packaging Authority instead recommends using the ``--index-url`` flag to limit the risk of this attack
    vector.

Next, you can install the nightly wheels using ``pip``. To do this, it is needed to specify the nightly index URL
through command-line flags, as showcased in the command below:

.. code-block:: bash

    pip install pybamm --pre --index-url # TODO: add index URL from CI job

Next, you can test that PyBaMM has been installed correctly by running the following command:

.. code-block:: bash

    python -c "import pybamm; print(pybamm.__version__)"


If you encounter any issues with the installation through this channel, please feel free to report them to us on
the `GitHub issue tracker <https://github.com/pybamm-team/PyBaMM/issues/new>`_.
