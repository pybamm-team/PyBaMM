# Contributing to PyBaMM

If you'd like to contribute to PyBaMM (thanks!), please have a look at the [guidelines below](#workflow).

If you're already familiar with our workflow, maybe have a quick look at the [pre-commit checks](#pre-commit-checks) directly below.

## Pre-commit checks

Before you commit any code, please perform the following checks:

- [No style issues](#coding-style-guidelines): `$ flake8`
- [All tests pass](#testing): `$ tox -e unit` (GNU/Linux and MacOS), `$ python -m tox -e windows-unit` (Windows)
- [The documentation builds](#building-the-documentation): `$ python -m tox -e docs`

## Workflow

We use [GIT](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### A. Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discussed before any coding is done.
2. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo (ideally on your own [fork](https://help.github.com/articles/fork-a-repo/)), where all changes will be made
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. [Install](https://pybamm.readthedocs.io/en/latest/install/install-from-source.html) PyBaMM with the developer options.
5. [Test](#testing) if your installation worked, using the test script: `$ python run-tests.py --unit`.

You now have everything you need to start making changes!

### B. Writing your code

5. PyBaMM is developed in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and makes heavy use of [NumPy](https://en.wikipedia.org/wiki/NumPy) (see also [NumPy for MatLab users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) and [Python for R users](http://blog.hackerearth.com/how-can-r-users-learn-python-for-data-science)).
6. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
7. Commit your changes to your branch with [useful, descriptive commit messages](https://chris.beams.io/posts/git-commit/): Remember these are publicly visible and should still make sense a few months ahead in time. While developing, you can keep using the GitHub issue you're working on as a place for discussion. [Refer to your commits](https://stackoverflow.com/questions/8910271/how-can-i-reference-a-commit-in-an-issue-comment-on-github) when discussing specific lines of code.
8. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### C. Merging your changes with PyBaMM

9. [Test your code!](#testing)
10. PyBaMM has online documentation at http://pybamm.readthedocs.io/. To make sure any new methods or classes you added show up there, please read the [documentation](#documentation) section.
11. If you added a major new feature, perhaps it should be showcased in an [example notebook](#example-notebooks).
12. When you feel your code is finished, or at least warrants serious discussion, run the [pre-commit checks](#pre-commit-checks) and then create a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) on [PyBaMM's GitHub page](https://github.com/pybamm-team/PyBaMM).
13. Once a PR has been created, it will be reviewed by any member of the community. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into PyBaMM main repository.

Finally, if you really, really, _really_ love developing PyBaMM, have a look at the current [project infrastructure](#infrastructure).


## Coding style guidelines

PyBaMM follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them.

### Flake8

We use [flake8](http://flake8.pycqa.org/en/latest/) to check our PEP8 adherence. To try this on your system, navigate to the PyBaMM directory in a console and type

```bash
flake8
```
Flake8 is configured inside the file `tox.ini`, under the section `[flake8]`, allowing us to ignore some errors. If you think this should be added or removed, please submit an [issue](#issues)

When you commit your changes they will be checked against flake8 automatically (see [infrastructure](#infrastructure)).


### Black

We use [black](https://black.readthedocs.io/en/stable/) to automatically configure our code to adhere to PEP8. Black can be used in two ways:

1. Command line: navigate to the PyBaMM directory in a console and type

```bash
black {source_file_or_directory}
```

2. Editor: black can be [configured](https://test-black.readthedocs.io/en/latest/editor_integration.html) to automatically reformat a python script each time the script is saved in an editor.

If you want to use black in your editor, you may need to change the max line length in your editor settings.

Even when code has been formatted by black, you should still make sure that it adheres to the PEP8 standard set by [Flake8](#flake8).

### Naming

Naming is hard. In general, we aim for descriptive class, method, and argument names. Avoid abbreviations when possible without making names overly long, so `mean` is better than `mu`, but a class name like `MyClass` is fine.

Class names are CamelCase, and start with an upper case letter, for example `MyOtherClass`. Method and variable names are lower case, and use underscores for word separation, for example `x` or `iteration_count`.


## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably sized download and an easy install_. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to PyBaMM users.
For these reasons, all dependencies in PyBaMM should be thought about carefully, and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their license permits it and is compatible with ours, but again should be considered carefully and discussed in the group. Snippets from blogs and [stackoverflow](https://stackoverflow.com/) can often be included without attribution, but if they solve a particularly nasty problem (or are very hard to read) it's often a good idea to attribute (and document) them, by making a comment with a link in the source code.

### Separating dependencies

On the other hand... We _do_ want to compare several tools, to generate documentation, and to speed up development. For this reason, the dependency structure is split into 4 parts:

1. Core PyBaMM: A minimal set, including things like NumPy, SciPy, etc. All infrastructure should run against this set of dependencies, as well as any numerical methods we implement ourselves.
2. Extras: Other inference packages and their dependencies. Methods we don't want to implement ourselves, but do want to provide an interface to can have their dependencies added here.
3. Documentation generating code: Everything you need to generate and work on the docs.
4. Development code: Everything you need to do PyBaMM development (so all of the above packages, plus flake8 and other testing tools).

Only 'core pybamm' is installed by default. The others have to be specified explicitly when running the installation command.

### Matplotlib

We use Matplotlib in PyBaMM, but with two caveats:

First, Matplotlib should only be used in plotting methods, and these should _never_ be called by other PyBaMM methods. So users who don't like Matplotlib will not be forced to use it in any way. Use in notebooks is OK and encouraged.

Second, Matplotlib should never be imported at the module level, but always inside methods. For example:

```
def plot_great_things(self, x, y, z):
    import matplotlib.pyplot as pl
    ...
```

This allows people to (1) use PyBaMM without ever importing Matplotlib and (2) configure Matplotlib's back-end in their scripts, which _must_ be done before e.g. `pyplot` is first imported.


## Testing

All code requires testing. We use the [unittest](https://docs.python.org/3.3/library/unittest.html) package for our tests. (These tests typically just check that the code runs without error, and so, are more _debugging_ than _testing_ in a strict sense. Nevertheless, they are very useful to have!)

If you have tox installed, to run unit tests, type

```bash
tox -e unit # (GNU/Linux and MacOS)
#
python -m tox -e windows-unit # (Windows)
```
else, type
```bash
python run-tests.py --unit
```

### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting this is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using any of these [assert methods](https://docs.python.org/3.3/library/unittest.html#assert-methods).

### Running more tests

The tests are divided into `unit` tests, whose aim is to check individual bits of code (e.g. discretising a gradient operator, or solving a simple ODE), and `integration` tests, which check how parts of the program interact as a whole (e.g. solving a full model).
If you want to check integration tests as well as unit tests, type

```bash
tox -e tests # (GNU/Linux and MacOS)
#
python -m tox -e windows-tests # (Windows)
```

When you commit anything to PyBaMM, these checks will also be run automatically (see [infrastructure](#infrastructure)).

### Testing notebooks

To test all example scripts and notebooks, type

```bash
tox -e examples # (GNU/Linux and MacOS)
#
python -m tox -e windows-examples # (Windows)
```

If notebooks fail because of changes to pybamm, it can be a bit of a hassle to debug. In these cases, you can create a temporary export of a notebook's Python content using

```bash
python run-tests.py --debook examples/notebooks/notebook-name.ipynb script.py
```

### Debugging

Often, the code you write won't pass the tests straight away, at which stage it will become necessary to debug.
The key to successful debugging is to isolate the problem by finding the smallest possible example that causes the bug.
In practice, there are a few tricks to help you to do this, which we give below.
Once you've isolated the issue, it's a good idea to add a unit test that replicates this issue, so that you can easily check whether it's been fixed, and make sure that it's easily picked up if it crops up again.
This also means that, if you can't fix the bug yourself, it will be much easier to ask for help (by opening a [bug-report issue](https://github.com/pybamm-team/PyBaMM/issues/new?template=bug_report.md)).

1. Run individual test scripts instead of the whole test suite:
```bash
python tests/unit/path/to/test
```
You can also run an individual test from a particular script, e.g.
```bash
python tests/unit/test_quick_plot.py TestQuickPlot.test_failure
```
If you want to run several, but not all, the tests from a script, you can restrict which tests are run from a particular script by using the skipping decorator:
```python
@unittest.skip("")
def test_bit_of_code(self):
    ...
```
or by just commenting out all the tests you don't want to run
2. Set break points, either in your IDE or using the python debugging module. To use the latter, add the following line where you want to set the break point
```python
import ipdb; ipdb.set_trace()
```
This will start the [Python interactive debugger](https://gist.github.com/mono0926/6326015). If you want to be able to use magic commands from `ipython`, such as `%timeit`, then set
```python
from IPython import embed; embed(); import ipdb; ipdb.set_trace()
```
at the break point instead.
Figuring out where to start the debugger is the real challenge. Some good ways to set debugging break points are:
  a. Try-except blocks. Suppose the line `do_something_complicated()` is raising a `ValueError`. Then you can put a try-except block around that line as:
  ```python
  try:
      do_something_complicated()
  except ValueError:
      import ipdb; ipdb.set_trace()
  ```
  This will start the debugger at the point where the `ValueError` was raised, and allow you to investigate further. Sometimes, it is more informative to put the try-except block further up the call stack than exactly where the error is raised.
  b. Warnings. If functions are raising warnings instead of errors, it can be hard to pinpoint where this is coming from. Here, you can use the `warnings` module to convert warnings to errors:
  ```python
  import warnings
  warnings.simplefilter("error")
  ```
  Then you can use a try-except block, as in a., but with, for example, `RuntimeWarning` instead of `ValueError`.
  c. Stepping through the expression tree. Most calls in PyBaMM are operations on [expression trees](https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/expression_tree/expression-tree.ipynb). To view an expression tree in ipython, you can use the `render` command:
  ```python
  expression_tree.render()
  ```
  You can then step through the expression tree, using the `children` attribute, to pinpoint exactly where a bug is coming from. For example, if `expression_tree.jac(y)` is failing, you can check `expression_tree.children[0].jac(y)`, then `expression_tree.children[0].children[0].jac(y)`, etc.
3. To isolate whether a bug is in a model, its jacobian or its simplified version, you can set the `use_jacobian` and/or `use_simplify` attributes of the model to `False` (they are both `True` by default for most models).
4. If a model isn't giving the answer you expect, you can try comparing it to other models. For example, you can investigate parameter limits in which two models should give the same answer by setting some parameters to be small or zero. The `StandardOutputComparison` class can be used to compare some standard outputs from battery models.
5. To get more information about what is going on under the hood, and hence understand what is causing the bug, you can set the [logging](https://realpython.com/python-logging/) level to `DEBUG` by adding the following line to your test or script:
```python3
pybamm.set_logging_level("DEBUG")
```
6. In models that inherit from `pybamm.BaseBatteryModel` (i.e. any battery model), you can use `self.process_parameters_and_discretise` to process a symbol and see what it will look like.

### Profiling

Sometimes, a bit of code will take much longer than you expect to run. In this case, you can set
```python
from IPython import embed; embed(); import ipdb; ipdb.set_trace()
```
as above, and then use some of the profiling tools. In order of increasing detail:
1. Simple timer. In ipython, the command
```
%time command_to_time()
```
tells you how long the line `command_to_time()` takes. You can use `%timeit` instead to run the command several times and obtain more accurate timings.
2. Simple profiler. Using `%prun` instead of `%time` will give a brief profiling report
3. Detailed profiler. You can install the detailed profiler `snakeviz` through pip:
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

These docstrings can be fairly simple, but can also make use of [reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html), a markup language designed specifically for writing [technical documentation](https://en.wikipedia.org/wiki/ReStructuredText). For example, you can link to other classes and methods by writing ```:class:`pybamm.Model` ``` and  ```:meth:`run()` ```.

In addition, we write a (very) small bit of documentation in separate reStructuredText files in the `docs` directory. Most of what these files do is simply import docstrings from the source code. But they also do things like add tables and indexes. If you've added a new class to a module, search the `docs` directory for that module's `.rst` file and add your class (in alphabetical order) to its index. If you've added a whole new module, copy-paste another module's file and add a link to your new file in the appropriate `index.rst` file.

Using [Sphinx](http://www.sphinx-doc.org/en/stable/) the documentation in `docs` can be converted to HTML, PDF, and other formats. In particular, we use it to generate the documentation on http://pybamm.readthedocs.io/

### Building the documentation

To test and debug the documentation, it's best to build it locally. To do this, navigate to your PyBaMM directory in a console, and then type:

```
python -m tox -e docs (GNU/Linux, MacOS and Windows)
```
And then visit the webpage served at http://127.0.0.1:8000. Each time a change to the documentation source is detected, the HTML is rebuilt and the browser automatically reloaded.

### Example notebooks

Major PyBaMM features are showcased in [Jupyter notebooks](https://jupyter.org/) stored in the [examples directory](examples/notebooks). Which features are "major" is of course wholly subjective, so please discuss on GitHub first!

All example notebooks should be listed in [examples/README.md](https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/README.md). Please follow the (naming and writing) style of existing notebooks where possible.

All the notebooks are tested daily.

## Citations

We aim to recognize all contributions by automatically generating citations to the relevant papers on which different parts of the code are built.
These will change depending on what models and solvers you use.
Adding the command

```python3
pybamm.print_citations()
```

to the end of a script will print all citations that were used by that script. This will print bibtex information to the terminal; passing a filename to `print_citations` will print the bibtex information to the specified file instead.

When you contribute code to PyBaMM, you can add your own papers that you would like to be cited if that code is used. First, add the bibtex for your paper to [CITATIONS.txt](pybamm/CITATIONS.txt). Then, add the line

```python3
pybamm.citations.register("your_paper_bibtex_identifier")
```

wherever code is called that uses that citation (for example, in functions or in the `__init__` method of a class such as a model or solver).

## Benchmarks

A benchmark suite is located in the `benchmarks` directory at the root of the PyBaMM project. These benchmarks can be run using [airspeed velocity](https://asv.readthedocs.io/en/stable/) (`asv`).

### Running the benchmarks
First of all, you'll need `asv` installed:
```shell
pip install asv
```

To run the benchmarks for the latest commit on the `develop` branch, simply enter the following command:
```shell
asv run
```
If it is the first time you run `asv`, you will be prompted for information about your machine (e.g. its name, operating system, architecture...).

Running the benchmarks can take a while, as all benchmarks are repeated several times to ensure statistically significant results. If accuracy isn't an issue, use the `--quick` option to avoid repeating each benchmark multiple times.
```shell
asv run --quick
```

Benchmarks can also be run over a range of commits. For instance, the following command runs the benchmark suite over every commit between version `0.3` and the tip of the `develop` branch:
```shell
asv run v0.3..develop
```
Further information on how to run benchmarks with `asv` can be found in the documentation at [Using airspeed velocity](https://asv.readthedocs.io/en/stable/using.html).

`asv` is configured using a file `asv.conf.json` located at the root of the PyBaMM repository. See the [asv reference](https://asv.readthedocs.io/en/stable/reference.html) for details on available settings and options.

Benchmark results are stored in a directory `results/` at the location of the configuration file. There is one result file per commit, per machine.

### Visualising benchmark results

`asv` is able to generate a static website with a visualisation of the benchmarks results, i.e. the benchmark's duration as a function of the commit hash.
To generate the website, use
```shell
asv publish
```
then, to view the website:
```shell
asv preview
```

Current benchmarks over PyBaMM's history can be viewed at https://pybamm-team.github.io/pybamm-bench/

### Adding benchmarks

To contribute benchmarks to PyBaMM, add a new benchmark function in one of the files in the `benchmarks/` directory.
Benchmarks are distributed across multiple files, grouped by theme. You're welcome to add a new file if none of your benchmarks fit into one of the already existing files.
Inside a benchmark file (e.g. `benchmarks/benchmarks.py`) benchmarks functions are grouped within classes.

Note that benchmark functions _must_ start with the prefix `time_`, for instance
```python3
def time_solve_SPM_ScipySolver(self):
        solver = pb.ScipySolver()
        solver.solve(self.model, [0, 3600])
```

In the case where some setup is necessary, but should not be timed, a `setup` function
can be defined as a method of the relevant class. For example:
```python3
class TimeSPM:
    def setup(self):
        model = pb.lithium_ion.SPM()
        geometry = model.default_geometry

	    # ...

        self.model = model

    def time_solve_SPM_ScipySolver(self):
        solver = pb.ScipySolver()
        solver.solve(self.model, [0, 3600])
```

Similarly, a `teardown` method will be run after the benchmark. Note that, unless the `--quick` option is used, benchmarks are executed several times for accuracy, and both the `setup` and `teardown` function are executed before/after each repetition.

Running benchmarks can take a while, and by default encountered exceptions will not be shown. When developing benchmarks, it is often convenient to use the following command instead of `asv run`:
```shell
asv dev
```

`asv dev` implies options `--quick`, `--show-stderr`, and `--dry-run` (to avoid updating the `results` directory).

## Infrastructure

### Setuptools

Installation of PyBaMM _and dependencies_ is handled via [setuptools](http://setuptools.readthedocs.io/)

Configuration files:

```
setup.py
```

Note that this file must be kept in sync with the version number in [pybamm/__init__.py](pybamm/__init__.py).

### Continuous Integration using GitHub actions

Each change pushed to the PyBaMM GitHub repository will trigger the test and benchmark suites to be run, using [GitHub actions](https://github.com/features/actions).

Tests are run for different operating systems, and for all python versions officially supported by PyBaMM. If you opened a Pull Request, feedback is directly available on the corresponding page. If all tests pass, a green tick will be displayed next to the corresponding test run. If one or more test(s) fail, a red cross will be displayed instead.

Similarly, the benchmark suite is automatically run for the most recently pushed commit. Benchmark results are compared to the results available for the latest commit on the `develop` branch. Should any significant performance regression be found, a red cross will be displayed next to the benchmark run.

In all cases, more details can be obtained by clicking on a specific run.

Configuration files for various GitHub actions workflow can be found in `.github/worklfows`.

### Codecov

Code coverage (how much of our code is actually seen by the (linux) unit tests) is tested using [Codecov](https://docs.codecov.io/), a report is visible on https://codecov.io/gh/pybamm-team/PyBaMM.

Configuration files:

```
tox.ini
```

### Read the Docs

Documentation is built using https://readthedocs.org/ and published on http://pybamm.readthedocs.io/.

### Google Colab

Editable notebooks are made available using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) [here](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/develop/).

### GitHub

GitHub does some magic with particular filenames. In particular:

- The first page people see when they go to [our GitHub page](https://github.com/pybamm-team/PyBaMM) displays the contents of [README.md](README.md), which is written in the [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) format. Some guidelines can be found [here](https://help.github.com/articles/about-readmes/).
- The license for using PyBaMM is stored in [LICENSE](LICENSE), and [automatically](https://help.github.com/articles/adding-a-license-to-a-repository/) linked to by GitHub.
- This file, [CONTRIBUTING.md](CONTRIBUTING.md) is recognised as the contribution guidelines and a link is [automatically](https://github.com/blog/1184-contributing-guidelines) displayed when new issues or pull requests are created.

## Acknowledgements

This CONTRIBUTING.md file, along with large sections of the code infrastructure,
was copied from the excellent [Pints GitHub repo](https://github.com/pints-team/pints)
