import os
import subprocess
try:
    import setuptools
except ImportError:
    from distutils.core import setup, find_packages
from distutils.cmd import Command

class BuildSundials(Command):
    """ A custom command to compile the SUNDIALS library as part of the PyBaMM
        installation process.
    """

    description = 'Compiles the SUNDIALS library.'
    user_options = [
        # The format is (long option, short option, description).
        ('sundials-dir=', None, 'relative path to sundials source dir')
    ]
    pybamm_dir = os.path.abspath(os.path.dirname(__file__))
    build_temp = 'build_sundials'
    install_dir = os.path.join(pybamm_dir,'sundials')

    def initialize_options(self):
        """Set default values for option(s)"""
        # Each user option is listed here with its default value.
        self.sundials_dir = os.path.join(self.pybamm_dir,'sundials-3.1.1')

    def finalize_options(self):
        """Post-process options"""
        if self.sundials_dir:
            CMakeLists=os.path.join(self.sundials_dir,'CMakeLists.txt')
            assert os.path.exists(CMakeLists), ('Could not find {}.'.format(CMakeLists))

    def _update_LD_LIBRARY_PATH(self):
        """ Look for current virtual python env and add export statement for LD_LIBRARY_PATH
        in activate script.
        If no virtual env found, then the current user's .bashrc file is modified instead.
        """
        export_statement='export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH'.format(self.install_dir)
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            script_path = os.path.join(venv_path, 'bin/activate')
        else:
            script_path = os.path.join(os.environ.get('HOME'), '.bashrc')

        if '{}/lib'.format(self.install_dir) in os.environ['LD_LIBRARY_PATH']:
            print("{}/lib was found in LD_LIBRARY_PATH.".format(self.install_dir))
            print("--> Not updating venv activate or .bashrc scripts")
        else:
            with open(script_path, 'a+') as fh:
                if not export_statement in fh.read():
                    fh.write(export_statement)
                    print("Adding {}/lib to LD_LIBRARY_PATH"
                          " in {}".format(self.install_dir, script_path))

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DLAPACK_ENABLE=ON' + build_directory,
            '-DSUNDIALS_INDEX_TYPE=int32_t',
            '-DBUILD_ARKODE:BOOL=OFF',
            '-DEXAMPLES_ENABLE:BOOL=OFF',
            '-DCMAKE_INSTALL_PREFIX=' + self.install_dir,
        ]

        if not os.path.exists(self.build_temp):
            print('-'*10, 'Creating build dir', '-'*40)
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        print('-'*10, 'Running CMake prepare', '-'*40)
        subprocess.run(['cmake', self.sundials_dir] + cmake_args,
                              cwd=self.build_temp)

        print('-'*10, 'Building the sundials', '-'*40)
        make_cmd = ['make', 'install']
        subprocess.run(make_cmd, cwd=self.build_temp)

        self._update_LD_LIBRARY_PATH()


# Load text for description and license
with open("README.md") as f:
    readme = f.read()

# Read version number from file
def load_version():
    try:
        import os

        root = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(root, "pybamm", "version"), "r") as f:
            version = f.read().strip().split(",")
        return ".".join([str(int(x)) for x in version])
    except Exception as e:
        raise RuntimeError("Unable to read version number (" + str(e) + ").")

setuptools.setup(
    cmdclass = {
        'build_sundials': BuildSundials,
    },
    name="pybamm",
    version=load_version()+".post4",
    description="Python Battery Mathematical Modelling.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/pybamm-team/PyBaMM",
    include_package_data=True,
    packages=setuptools.find_packages(include=("pybamm", "pybamm.*")),
    package_data={
        "pybamm": [
            "./version",
            "../input/parameters/lithium-ion/*.csv",
            "../input/parameters/lithium-ion/*.py",
            "../input/parameters/lead-acid/*.csv",
            "../input/parameters/lead-acid/*.py",
        ]
    },
    # List of dependencies
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.0",
        "pandas>=0.23",
        "anytree>=2.4.3",
        "autograd>=1.2",
        "scikit-fem>=0.2.0",
        # Note: Matplotlib is loaded for debug plots, but to ensure pybamm runs
        # on systems without an attached display, it should never be imported
        # outside of plot() methods.
        # Should not be imported
        "matplotlib>=2.0",
    ],
    extras_require={
        "docs": ["sphinx>=1.5", "guzzle-sphinx-theme"],  # For doc generation
        "dev": [
            "flake8>=3",  # For code style checking
            "black",  # For code style auto-formatting
            "jupyter",  # For documentation and testing
        ],
    },
)
