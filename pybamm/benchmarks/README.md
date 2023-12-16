## Benchmarks

This directory contains the benchmark suites of the PyBaMM project. These benchmarks can be run using [airspeed velocity](https://asv.readthedocs.io/en/stable/) (`asv`).

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
