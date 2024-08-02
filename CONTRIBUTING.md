# Contributing to PyBaMM

If you'd like to contribute to PyBaMM (thanks!), please have a look at the [guidelines below](#workflow).

If you're already familiar with our workflow, maybe have a quick look at the [pre-commit checks](#pre-commit-checks) directly below.

## Pre-commit checks

Before you commit any code, please perform the following checks:

- [All tests pass](#testing): `$ nox -s unit`
- [The documentation builds](#building-the-documentation): `$ nox -s docs`

### Installing and using pre-commit

`PyBaMM` uses a set of `pre-commit` hooks and the `pre-commit` bot to format and prettify the codebase. The hooks can be installed locally using -

```bash
pip install pre-commit
pre-commit install
```

This would run the checks every time a commit is created locally. The checks will only run on the files modified by that commit, but the checks can be triggered for all the files using -

```bash
pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

## Workflow

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### A. Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discussed before any coding is done.
2. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo (ideally on your own [fork](https://help.github.com/articles/fork-a-repo/)), where all changes will be made
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. [Install](https://docs.pybamm.org/en/latest/source/user_guide/installation/install-from-source.html) PyBaMM with the developer options.
5. [Test](#testing) if your installation worked, using the test script: `$ python run-tests.py --unit`.

You now have everything you need to start making changes!

### B. Writing your code

6. PyBaMM is developed in [Python](https://www.python.org)), and makes heavy use of [NumPy](https://numpy.org/) (see also [NumPy for MatLab users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) and [Python for R users](https://www.rebeccabarter.com/blog/2023-09-11-from_r_to_python)).
7. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
8. Commit your changes to your branch with [useful, descriptive commit messages](https://chris.beams.io/posts/git-commit/): Remember these are
   publicly visible and should still make sense a few months ahead in time.
   While developing, you can keep using the GitHub issue you're working on
   as a place for discussion.
9. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### C. Merging your changes with PyBaMM

10. [Test your code!](#testing)
11. PyBaMM has online documentation at http://docs.pybamm.org/. To make sure any new methods or classes you added show up there, please read the [documentation](#documentation) section.
12. If you added a major new feature, perhaps it should be showcased in an [example notebook](#example-notebooks).
13. When you feel your code is finished, or at least warrants serious discussion, run the [pre-commit checks](#pre-commit-checks) and then create a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) on [PyBaMM's GitHub page](https://github.com/pybamm-team/PyBaMM).
14. Once a PR has been created, it will be reviewed by any member of the community. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into PyBaMM main repository.

Finally, if you really, really, _really_ love developing PyBaMM, have a look at the current [project infrastructure](#infrastructure).

## Coding style guidelines

PyBaMM follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them. We recommend using pre-commit hooks to check your code before committing it. See [installing and using pre-commit](#installing-and-using-pre-commit) section for more details.

### Ruff

We use [ruff](https://github.com/charliermarsh/ruff) to check our PEP8 adherence. To try this on your system, navigate to the PyBaMM directory in a console and type

```bash
python -m pip install pre-commit
pre-commit run ruff
```

ruff is configured inside the file `pre-commit-config.yaml`, allowing us to ignore some errors. If you think this should be added or removed, please submit an [issue](https://github.com/pybamm-team/PyBaMM/issues)

When you commit your changes they will be checked against ruff automatically (see [Pre-commit checks](#pre-commit-checks)).

### Naming

Naming is hard. In general, we aim for descriptive class, method, and argument names. Avoid abbreviations when possible without making names overly long, so `mean` is better than `mu`, but a class name like `MyClass` is fine.

Class names are CamelCase, and start with an upper case letter, for example `MyOtherClass`. Method and variable names are lower case, and use underscores for word separation, for example `x` or `iteration_count`.

## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably sized download and an easy install_. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to PyBaMM users.
For these reasons, all dependencies in PyBaMM should be thought about carefully, and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their
license permits it and is compatible with ours, but again should be
considered carefully and discussed in the group. Snippets from blogs and
stackoverflow can often be included without attribution, but if they solve a
particularly nasty problem (or are very hard to read) it's often a good idea to
attribute (and document) them, by making a comment with a link in the source code.

### Separating dependencies

On the other hand... We _do_ want to compare several tools, to generate documentation, and to speed up development. For this reason, the dependency structure is split into 4 parts:

1. Core PyBaMM: A minimal set, including things like NumPy, SciPy, etc. All infrastructure should run against this set of dependencies, as well as any numerical methods we implement ourselves.
2. Extras: Other inference packages and their dependencies. Methods we don't want to implement ourselves, but do want to provide an interface to can have their dependencies added here.
3. Documentation generating code: Everything you need to generate and work on the docs.
4. Development code: Everything you need to do PyBaMM development (so all of the above packages, plus ruff and other testing tools).

Only 'core pybamm' is installed by default. The others have to be specified explicitly when running the installation command.

### Managing Optional Dependencies and Their Imports

PyBaMM utilizes optional dependencies to allow users to choose which additional libraries they want to use. Managing these optional dependencies and their imports is essential to provide flexibility to PyBaMM users.

PyBaMM provides a utility function `import_optional_dependency`, to check for the availability of optional dependencies within methods. This function can be used to conditionally import optional dependencies only if they are available. Here's how to use it:

Optional dependencies should never be imported at the module level, but always inside methods. For example:

```
def use_pybtex(x,y,z):
    pybtex = import_optional_dependency("pybtex")
    ...
```

While importing a specific module instead of an entire package/library:

```python
def use_parse_file(x, y, z):
    parse_file = import_optional_dependency("pybtex.database", "parse_file")
    ...
```

This allows people to (1) use PyBaMM without importing optional dependencies by default and (2) configure module-dependent functionalities in their scripts, which _must_ be done before e.g. `print_citations` method is first imported.

**Writing Tests for Optional Dependencies**

Below, we list the currently available test functions to provide an overview. If you find it useful to add new test cases please do so within `tests/unit/test_util.py`.

Currently, there are three functions to test what concerns optional dependencies:
- `test_import_optional_dependency`
- `test_pybamm_import`
- `test_optional_dependencies`

The `test_import_optional_dependency` function extracts the optional dependencies installed in the setup environment, makes them unimportable (by setting them to `None` among the `sys.modules`), and tests that the `pybamm.util.import_optional_dependency` function throws a `ModuleNotFoundError` exception when their import is attempted.

The `test_pybamm_import` function extracts the optional dependencies installed in the setup environment and makes them unimportable (by setting them to `None` among the `sys.modules`), unloads `pybamm` and its sub-modules, and finally tests that `pybamm` can be imported successfully. In fact, it is essential that the `pybamm` package is importable with only the mandatory dependencies.

The `test_optional_dependencies` function extracts `pybamm` mandatory distribution packages and verifies that they are not present in the optional distribution packages list in `pyproject.toml`. This test is crucial for ensuring the consistency of the released package information and potential updates to dependencies during development.

## Testing

All code requires testing. We use the [pytest](https://docs.pytest.org/en/stable/) package for our tests. (These tests typically just check that the code runs without error, and so, are more _debugging_ than _testing_ in a strict sense. Nevertheless, they are very useful to have!)

We use following plugins for various needs:

[nbmake](https://github.com/treebeardtech/nbmake/) : plugins to test the example notebooks.

[pytest-xdist](https://pypi.org/project/pytest-xdist/) : plugins to run tests in parallel.

If you have `nox` installed, to run unit tests, type

```bash
nox -s unit
```

else, type

```bash
pytest -m unit
```

### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting this is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using [assert statements](https://docs.pytest.org/en/7.1.x/how-to/assert.html).

### Running more tests

The tests are divided into `unit` tests, whose aim is to check individual bits of code (e.g. discretising a gradient operator, or solving a simple ODE), and `integration` tests, which check how parts of the program interact as a whole (e.g. solving a full model).
If you want to check integration tests as well as unit tests, type

```bash
nox -s tests
```

When you commit anything to PyBaMM, these checks will also be run automatically (see [infrastructure](#infrastructure)).

### Testing the example notebooks

To test all the example notebooks in the `docs/source/examples/` folder with `pytest` and `nbmake`, type

```bash
nox -s examples
```

Alternatively, you may use `pytest` directly with the `--nbmake` flag:

```bash
pytest --nbmake docs/source/examples/
```

which runs all the notebooks in the `docs/source/examples/notebooks/` folder in parallel by default, using the `pytest-xdist` plugin.

Sometimes, debugging a notebook can be a hassle. To run a single notebook, pass the path to it to `pytest`:

```bash
pytest --nbmake docs/source/examples/notebooks/notebook-name.ipynb
```

or, alternatively, you can use posargs to pass the path to the notebook to `nox`. For example:

```bash
nox -s examples -- docs/source/examples/notebooks/notebook-name.ipynb
```

You may also test multiple notebooks this way. Passing the path to a folder will run all the notebooks in that folder:

```bash
nox -s examples -- docs/source/examples/notebooks/models/
```

You may also use an appropriate [glob pattern](https://docs.python.org/3/library/glob.html) to run all notebooks matching a particular folder or name pattern.

To edit the structure and how the Jupyter notebooks get rendered in the Sphinx documentation (using `nbsphinx`), install [Pandoc](https://pandoc.org/installing.html) on your system, either using `conda` (through the `conda-forge` channel)

```bash
conda install -c conda-forge pandoc
```

or refer to the [Pandoc installation instructions](https://pandoc.org/installing.html) specific to your platform.

### Testing the example scripts

To test all the example scripts in the `examples/` folder, type

```bash
nox -s scripts
```

### Debugging

Often, the code you write won't pass the tests straight away, at which stage it will become necessary to debug.
The key to successful debugging is to isolate the problem by finding the smallest possible example that causes the bug.
In practice, there are a few tricks to help you to do this, which we give below.
Once you've isolated the issue, it's a good idea to add a unit test that replicates this issue, so that you can easily check whether it's been fixed, and make sure that it's easily picked up if it crops up again.
This also means that, if you can't fix the bug yourself, it will be much easier to ask for help (by opening a [bug-report issue](https://github.com/pybamm-team/PyBaMM/issues/new?template=bug_report.md)).

1. Run individual test scripts instead of the whole test suite:

   ```bash
   pytest tests/unit/path/to/test
   ```

   You can also run an individual test from a particular script, e.g.

   ```bash
   pytest tests/unit/test_plotting/test_quick_plot.py::TestQuickPlot::test_simple_ode_model
   ```

   If you want to run several, but not all, the tests from a script, you can restrict which tests are run from a particular script by using the skipping decorator:

   ```python
   @pytest.mark.skip("")
   def test_bit_of_code(self):
       ...
   ```

   or by just commenting out all the tests you don't want to run.
2. Set break points, either in your IDE or using the Python debugging module. To use the latter, add the following line where you want to set the break point

   ```python
   import ipdb

   ipdb.set_trace()
   ```

   This will start the [Python interactive debugger](https://gist.github.com/mono0926/6326015). If you want to be able to use magic commands from `ipython`, such as `%timeit`, then set

   ```python
   from IPython import embed

   embed()
   import ipdb

   ipdb.set_trace()
   ```

   at the break point instead.
   Figuring out where to start the debugger is the real challenge. Some good ways to set debugging break points are:

   1. Try-except blocks. Suppose the line `do_something_complicated()` is raising a `ValueError`. Then you can put a try-except block around that line as:

      ```python
      try:
          do_something_complicated()
      except ValueError:
          import ipdb

          ipdb.set_trace()
      ```

      This will start the debugger at the point where the `ValueError` was raised, and allow you to investigate further. Sometimes, it is more informative to put the try-except block further up the call stack than exactly where the error is raised.
   2. Warnings. If functions are raising warnings instead of errors, it can be hard to pinpoint where this is coming from. Here, you can use the `warnings` module to convert warnings to errors:

      ```python
      import warnings

      warnings.simplefilter("error")
      ```

      Then you can use a try-except block, as in a., but with, for example, `RuntimeWarning` instead of `ValueError`.
   3. Stepping through the expression tree. Most calls in PyBaMM are operations on [expression trees](https://github.com/pybamm-team/PyBaMM/blob/develop/docs/source/examples/notebooks/expression_tree/expression-tree.ipynb). To view an expression tree in ipython, you can use the `render` command:

      ```python
      expression_tree.render()
      ```

      You can then step through the expression tree, using the `children` attribute, to pinpoint exactly where a bug is coming from. For example, if `expression_tree.jac(y)` is failing, you can check `expression_tree.children[0].jac(y)`, then `expression_tree.children[0].children[0].jac(y)`, etc.
3. To isolate whether a bug is in a model, its Jacobian or its simplified version, you can set the `use_jacobian` and/or `use_simplify` attributes of the model to `False` (they are both `True` by default for most models).
4. If a model isn't giving the answer you expect, you can try comparing it to other models. For example, you can investigate parameter limits in which two models should give the same answer by setting some parameters to be small or zero. The `StandardOutputComparison` class can be used to compare some standard outputs from battery models.
5. To get more information about what is going on under the hood, and hence understand what is causing the bug, you can set the [logging](https://realpython.com/python-logging/) level to `DEBUG` by adding the following line to your test or script:

   ```python3
   pybamm.set_logging_level("DEBUG")
   ```
6. In models that inherit from `pybamm.BaseBatteryModel` (i.e. any battery model), you can use `self.process_parameters_and_discretise` to process a symbol and see what it will look like.

### Profiling

Sometimes, a bit of code will take much longer than you expect to run. In this case, you can set

```python
from IPython import embed

embed()
import ipdb

ipdb.set_trace()
```

as above, and then use some of the profiling tools. In order of increasing detail:

1. Simple timer. In ipython, the command

   ```
   %time command_to_time()
   ```

   tells you how long the line `command_to_time()` takes. You can use `%timeit` instead to run the command several times and obtain more accurate timings.
2. Simple profiler. Using `%prun` instead of `%time` will give a brief profiling report 3. Detailed profiler. You can install the detailed profiler `snakeviz` through pip:

   ```bash
   pip install snakeviz
   ```

   and then, in ipython, run

   ```
   %load_ext snakeviz
   %snakeviz command_to_time()
   ```

   This will open a window in your browser with detailed profiling information.

## Documentation

PyBaMM is documented in several ways.

First and foremost, every method and every class should have a [docstring](https://www.python.org/dev/peps/pep-0257/) that describes in plain terms what it does, and what the expected input and output is.

These docstrings can be fairly simple, but can also make use of [reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html), a markup language designed specifically for writing [technical documentation](https://en.wikipedia.org/wiki/ReStructuredText). For example, you can link to other classes and methods by writing ``:class:`pybamm.Model` `` and ``:meth:`run()` ``.

In addition, we write a (very) small bit of documentation in separate reStructuredText files in the `docs` directory. Most of what these files do is simply import docstrings from the source code. But they also do things like add tables and indexes. If you've added a new class to a module, search the `docs` directory for that module's `.rst` file and add your class (in alphabetical order) to its index. If you've added a whole new module, copy-paste another module's file and add a link to your new file in the appropriate `index.rst` file.

Using [Sphinx](http://www.sphinx-doc.org/en/stable/) the documentation in `docs` can be converted to HTML, PDF, and other formats. In particular, we use it to generate the documentation on http://docs.pybamm.org/

### Building the documentation

To test and debug the documentation, it's best to build it locally. To do this, navigate to your PyBaMM directory in a console, and then type (on GNU/Linux, macOS, and Windows):

```
nox -s docs
```

And then visit the webpage served at `http://127.0.0.1:8000`. Each time a change to the documentation source is detected, the HTML is rebuilt and the browser automatically reloaded. In CI, the docs are built and tested using the `docs` session in the `noxfile.py` file with warnings turned into errors, to fail the build. The warnings can be removed or ignored by adding the appropriate warning identifier to the `suppress_warnings` list in `docs/conf.py`.

### Example notebooks

Major PyBaMM features are showcased in [Jupyter notebooks](https://jupyter.org/) stored in the [docs/source/examples directory](https://github.com/pybamm-team/PyBaMM/tree/develop/docs/source/examples). Which features are "major" is of course wholly subjective, so please discuss on GitHub first!

All example notebooks should be listed in [docs/source/examples/index.rst](https://github.com/pybamm-team/PyBaMM/blob/develop/docs/source/examples/index.rst). Please follow the (naming and writing) style of existing notebooks where possible.

All the notebooks are tested daily.

## Citations

We aim to recognize all contributions by automatically generating citations to the relevant papers on which different parts of the code are built.
These will change depending on what models and solvers you use.
Adding the command

```python3
pybamm.print_citations()
```

to the end of a script will print all citations that were used by that script. This will print BibTeX information to the terminal; passing a filename to `print_citations` will print the BibTeX information to the specified file instead.

When you contribute code to PyBaMM, you can add your own papers that you would like to be cited if that code is used. First, add the BibTeX for your paper to [CITATIONS.bib](https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/CITATIONS.bib). Then, add the line

```python3
pybamm.citations.register("your_paper_bibtex_identifier")
```

wherever code is called that uses that citation (for example, in functions or in the `__init__` method of a class such as a model or solver).

## Infrastructure

### Installation

Installation of PyBaMM and its dependencies is handled via [pip](https://pip.pypa.io/en/stable/) and [setuptools](http://setuptools.readthedocs.io/). It uses `CMake` to compile C++ extensions using [`pybind11`](https://pybind11.readthedocs.io/en/stable/) and [`casadi`](https://web.casadi.org/). The installation process is described in detail in the [source installation](https://docs.pybamm.org/en/latest/source/user_guide/installation/install-from-source.html) page and is configured through the `CMakeLists.txt` file.

Configuration files:

```
setup.py
pyproject.toml
MANIFEST.in
```

Note: `MANIFEST.in` is used to include and exclude non-Python files and auxiliary package data for PyBaMM when distributing it. If a file is not included in `MANIFEST.in`, it will not be included in the source distribution (SDist) and subsequently not be included in the binary distribution (wheel).

### Continuous Integration using GitHub Actions

Each change pushed to the PyBaMM GitHub repository will trigger the test and benchmark suites to be run, using [GitHub Actions](https://github.com/features/actions).

Tests are run for different operating systems, and for all Python versions officially supported by PyBaMM. If you opened a Pull Request, feedback is directly available on the corresponding page. If all tests pass, a green tick will be displayed next to the corresponding test run. If one or more test(s) fail, a red cross will be displayed instead.

Similarly, the benchmark suite is automatically run for the most recently pushed commit. Benchmark results are compared to the results available for the latest commit on the `develop` branch. Should any significant performance regression be found, a red cross will be displayed next to the benchmark run.

In all cases, more details can be obtained by clicking on a specific run.

Configuration files for various GitHub actions workflow can be found in `.github/worklfows`.

### Codecov

Code coverage (how much of our code is actually seen by the (linux) unit tests) is tested using [Codecov](https://docs.codecov.io/), a report is visible on https://codecov.io/gh/pybamm-team/PyBaMM.

Configuration files:

```
.coveragerc
```

### Read the Docs

Documentation is built using https://readthedocs.org/ and published on http://docs.pybamm.org/.

### Google Colab

Editable notebooks are made available using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) [here](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/).

### GitHub

GitHub does some magic with particular filenames. In particular:

- The first page people see when they go to [our GitHub page](https://github.com/pybamm-team/PyBaMM) displays the contents of [README.md](https://github.com/pybamm-team/PyBaMM/blob/develop/README.md), which is written in the [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) format. Some guidelines can be found [here](https://help.github.com/articles/about-readmes/).
- The license for using PyBaMM is stored in [LICENSE](https://github.com/pybamm-team/PyBaMM/blob/develop/LICENSE.txt), and [automatically](https://help.github.com/articles/adding-a-license-to-a-repository/) linked to by GitHub.
- This file, [CONTRIBUTING.md](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md) is recognised as the contribution guidelines and a link is [automatically](https://github.com/blog/1184-contributing-guidelines) displayed when new issues or pull requests are created.

## Acknowledgements

This CONTRIBUTING.md file, along with large sections of the code infrastructure,
was copied from the excellent [Pints GitHub repo](https://github.com/pints-team/pints)
