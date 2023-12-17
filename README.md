![PyBaMM_logo](https://user-images.githubusercontent.com/20817509/107091287-8ad46a80-67cf-11eb-86f5-7ebef7c72a1e.png)

#

<div align="center">

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](http://numfocus.org)
[![Scheduled](https://github.com/pybamm-team/PyBaMM/actions/workflows/run_periodic_tests.yml/badge.svg?branch=develop)](https://github.com/pybamm-team/PyBaMM/actions/workflows/run_periodic_tests.yml)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://docs.pybamm.org/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/main/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/)
[![DOI](https://zenodo.org/badge/DOI/10.5334/jors.309.svg)](https://doi.org/10.5334/jors.309)
[![release](https://img.shields.io/github/v/release/pybamm-team/PyBaMM?color=yellow)](https://github.com/pybamm-team/PyBaMM/releases)
[![code style](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-71-orange.svg)](#-contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

</div>

# PyBaMM

PyBaMM (Python Battery Mathematical Modelling) is an open-source battery simulation package
written in Python. Our mission is to accelerate battery modelling research by
providing open-source tools for multi-institutional, interdisciplinary collaboration.
Broadly, PyBaMM consists of
(i) a framework for writing and solving systems
of differential equations,
(ii) a library of battery models and parameters, and
(iii) specialized tools for simulating battery-specific experiments and visualizing the results.
Together, these enable flexible model definitions and fast battery simulations, allowing users to
explore the effect of different battery designs and modeling assumptions under a variety of operating scenarios.

[//]: # "numfocus-fiscal-sponsor-attribution"

PyBaMM uses an [open governance model](https://pybamm.org/governance/)
and is fiscally sponsored by [NumFOCUS](https://numfocus.org/). Consider making
a [tax-deductible donation](https://numfocus.org/donate-for-pybamm) to help the project
pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
  <a href="https://numfocus.org/project/pybamm">
    <img height="60px"
         src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
         align="center">
  </a>
</div>
<br>

## 💻 Using PyBaMM

The easiest way to use PyBaMM is to run a 1C constant-current discharge with a model of your choice with all the default settings:

```python3
import pybamm

model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()
```

or simulate an experiment such as a constant-current discharge followed by a constant-current-constant-voltage charge:

```python3
import pybamm

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour",
        )
    ]
    * 3,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
```

However, much greater customisation is available. It is possible to change the physics, parameter values, geometry, submesh type, number of submesh points, methods for spatial discretisation and solver for integration (see DFN [script](https://github.com/pybamm-team/PyBaMM/blob/develop/examples/scripts/DFN.py) or [notebook](https://github.com/pybamm-team/PyBaMM/blob/develop/docs/source/examples/notebooks/models/DFN.ipynb)).

For new users we recommend the [Getting Started](https://github.com/pybamm-team/PyBaMM/tree/develop/docs/source/examples/notebooks/getting_started/) guides. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM, and can either be downloaded and used locally, or used online through [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/).

Further details can be found in a number of [detailed examples](https://github.com/pybamm-team/PyBaMM/tree/develop/examples), hosted here on
github. In addition, there is a [full API documentation](https://docs.pybamm.org/en/latest/source/api/index.html),
hosted on [Read The Docs](https://readthedocs.org/).
Additional supporting material can be found
[here](https://github.com/pybamm-team/pybamm-supporting-material/).

Note that the examples on the default `develop` branch are tested on the latest `develop` commit. This may sometimes cause errors when running the examples on the pybamm pip package, which is synced to the `main` branch. You can switch to the `main` branch on github to see the version of the examples that is compatible with the latest pip release.

## Versioning

PyBaMM makes releases every four months and we use [CalVer](https://calver.org/), which means that the version number is `YY.MM`. The releases happen, approximately, at the end of January, May and September. There is no difference between releases that increment the year and releases that increment the month; in particular, releases that increment the month may introduce breaking changes. Breaking changes for each release are communicated via the [CHANGELOG](CHANGELOG.md), and come with deprecation warnings or errors that are kept for at least one year (3 releases). If you find a breaking change that is not documented, or think it should be undone, please open an issue on [GitHub](https://github.com/pybamm-team/pybamm).

## 🚀 Installing PyBaMM

PyBaMM is available on GNU/Linux, MacOS and Windows.
We strongly recommend to install PyBaMM within a python virtual environment, in order not to alter any distribution python files.
For instructions on how to create a virtual environment for PyBaMM, see [the documentation](https://docs.pybamm.org/en/latest/source/user_guide/installation/GNU-linux.html#user-install).

### Using pip

[![pypi](https://img.shields.io/pypi/v/pybamm?color=blue)](https://pypi.org/project/pybamm/)
[![downloads](https://img.shields.io/pypi/dm/pybamm?color=blue)](https://pypi.org/project/pybamm/)

```bash
pip install pybamm
```

### Using conda

PyBaMM is available as a conda package through the conda-forge channel.

[![conda_forge](https://img.shields.io/conda/vn/conda-forge/pybamm?color=green)](https://anaconda.org/conda-forge/pybamm)
[![downloads](https://img.shields.io/conda/dn/conda-forge/pybamm?color=green)](https://anaconda.org/conda-forge/pybamm)

```bash
conda install -c conda-forge pybamm
```

### Optional solvers

Following GNU/Linux and macOS solvers are optionally available:

- [scikits.odes](https://scikits-odes.readthedocs.io/en/latest/)-based solver, see [the documentation](https://docs.pybamm.org/en/latest/source/user_guide/installation/GNU-linux.html#optional-scikits-odes-solver).
- [jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)-based solver, see [the documentation](https://docs.pybamm.org/en/latest/source/user_guide/installation/GNU-linux.html#optional-jaxsolver).

## 📖 Citing PyBaMM

If you use PyBaMM in your work, please cite our paper

> Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., & Chapman, S. J. (2021). Python Battery Mathematical Modelling (PyBaMM). _Journal of Open Research Software, 9(1)_.

You can use the BibTeX

```
@article{Sulzer2021,
  title = {{Python Battery Mathematical Modelling (PyBaMM)}},
  author = {Sulzer, Valentin and Marquis, Scott G. and Timms, Robert and Robinson, Martin and Chapman, S. Jon},
  doi = {10.5334/jors.309},
  journal = {Journal of Open Research Software},
  publisher = {Software Sustainability Institute},
  volume = {9},
  number = {1},
  pages = {14},
  year = {2021}
}
```

We would be grateful if you could also cite the relevant papers. These will change depending on what models and solvers you use. To find out which papers you should cite, add the line

```python3
pybamm.print_citations()
```

to the end of your script. This will print BibTeX information to the terminal; passing a filename to `print_citations` will print the BibTeX information to the specified file instead. A list of all citations can also be found in the [citations file](https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/CITATIONS.bib). In particular, PyBaMM relies heavily on [CasADi](https://web.casadi.org/publications/).
See [CONTRIBUTING.md](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md#citations) for information on how to add your own citations when you contribute.

## 🛠️ Contributing to PyBaMM

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md) first.

## 📫 Get in touch

For any questions, comments, suggestions or bug reports, please see the [contact page](https://www.pybamm.org/contact).

## 📃 License

PyBaMM is fully open source. For more information about its license, see [LICENSE](https://github.com/pybamm-team/PyBaMM/blob/develop/LICENSE.txt).

## ✨ Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://sites.google.com/view/valentinsulzer"><img src="https://avatars3.githubusercontent.com/u/20817509?v=4?s=100" width="100px;" alt="Valentin Sulzer"/><br /><sub><b>Valentin Sulzer</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Atinosulzer" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tinosulzer" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tinosulzer" title="Documentation">📖</a> <a href="#example-tinosulzer" title="Examples">💡</a> <a href="#ideas-tinosulzer" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-tinosulzer" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Atinosulzer" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tinosulzer" title="Tests">⚠️</a> <a href="#tutorial-tinosulzer" title="Tutorials">✅</a> <a href="#blog-tinosulzer" title="Blogposts">📝</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.robertwtimms.com"><img src="https://avatars1.githubusercontent.com/u/43040151?v=4?s=100" width="100px;" alt="Robert Timms"/><br /><sub><b>Robert Timms</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Artimms" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=rtimms" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=rtimms" title="Documentation">📖</a> <a href="#example-rtimms" title="Examples">💡</a> <a href="#ideas-rtimms" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-rtimms" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Artimms" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=rtimms" title="Tests">⚠️</a> <a href="#tutorial-rtimms" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Scottmar93"><img src="https://avatars1.githubusercontent.com/u/22661308?v=4?s=100" width="100px;" alt="Scott Marquis"/><br /><sub><b>Scott Marquis</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3AScottmar93" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Scottmar93" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Scottmar93" title="Documentation">📖</a> <a href="#example-Scottmar93" title="Examples">💡</a> <a href="#ideas-Scottmar93" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-Scottmar93" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3AScottmar93" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Scottmar93" title="Tests">⚠️</a> <a href="#tutorial-Scottmar93" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/martinjrobins"><img src="https://avatars3.githubusercontent.com/u/1148404?v=4?s=100" width="100px;" alt="Martin Robinson"/><br /><sub><b>Martin Robinson</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Amartinjrobins" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=martinjrobins" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=martinjrobins" title="Documentation">📖</a> <a href="#example-martinjrobins" title="Examples">💡</a> <a href="#ideas-martinjrobins" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Amartinjrobins" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=martinjrobins" title="Tests">⚠️</a> <a href="#tutorial-martinjrobins" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.brosaplanella.com"><img src="https://avatars3.githubusercontent.com/u/28443643?v=4?s=100" width="100px;" alt="Ferran Brosa Planella"/><br /><sub><b>Ferran Brosa Planella</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Abrosaplanella" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Abrosaplanella" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=brosaplanella" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=brosaplanella" title="Documentation">📖</a> <a href="#example-brosaplanella" title="Examples">💡</a> <a href="#ideas-brosaplanella" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-brosaplanella" title="Maintenance">🚧</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=brosaplanella" title="Tests">⚠️</a> <a href="#tutorial-brosaplanella" title="Tutorials">✅</a> <a href="#blog-brosaplanella" title="Blogposts">📝</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TomTranter"><img src="https://avatars3.githubusercontent.com/u/7068741?v=4?s=100" width="100px;" alt="Tom Tranter"/><br /><sub><b>Tom Tranter</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3ATomTranter" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=TomTranter" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=TomTranter" title="Documentation">📖</a> <a href="#example-TomTranter" title="Examples">💡</a> <a href="#ideas-TomTranter" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3ATomTranter" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=TomTranter" title="Tests">⚠️</a> <a href="#tutorial-TomTranter" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://tlestang.github.io"><img src="https://avatars3.githubusercontent.com/u/13448239?v=4?s=100" width="100px;" alt="Thibault Lestang"/><br /><sub><b>Thibault Lestang</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Atlestang" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tlestang" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tlestang" title="Documentation">📖</a> <a href="#example-tlestang" title="Examples">💡</a> <a href="#ideas-tlestang" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Atlestang" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tlestang" title="Tests">⚠️</a> <a href="#infra-tlestang" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dalonsoa"><img src="https://avatars1.githubusercontent.com/u/6095790?v=4?s=100" width="100px;" alt="Diego"/><br /><sub><b>Diego</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Adalonsoa" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Adalonsoa" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=dalonsoa" title="Code">💻</a> <a href="#infra-dalonsoa" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/felipe-salinas"><img src="https://avatars2.githubusercontent.com/u/64426781?v=4?s=100" width="100px;" alt="felipe-salinas"/><br /><sub><b>felipe-salinas</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=felipe-salinas" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=felipe-salinas" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/suhaklee"><img src="https://avatars3.githubusercontent.com/u/57151989?v=4?s=100" width="100px;" alt="suhaklee"/><br /><sub><b>suhaklee</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=suhaklee" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=suhaklee" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/viviantran27"><img src="https://avatars0.githubusercontent.com/u/6379429?v=4?s=100" width="100px;" alt="viviantran27"/><br /><sub><b>viviantran27</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=viviantran27" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=viviantran27" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gyouhoc"><img src="https://avatars0.githubusercontent.com/u/60714526?v=4?s=100" width="100px;" alt="gyouhoc"/><br /><sub><b>gyouhoc</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Agyouhoc" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=gyouhoc" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=gyouhoc" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/YannickNoelStephanKuhn"><img src="https://avatars0.githubusercontent.com/u/62429912?v=4?s=100" width="100px;" alt="Yannick Kuhn"/><br /><sub><b>Yannick Kuhn</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=YannickNoelStephanKuhn" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=YannickNoelStephanKuhn" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://batterymodel.co.uk"><img src="https://avatars2.githubusercontent.com/u/39409226?v=4?s=100" width="100px;" alt="Jacqueline Edge"/><br /><sub><b>Jacqueline Edge</b></sub></a><br /><a href="#ideas-jedgedrudd" title="Ideas, Planning, & Feedback">🤔</a> <a href="#eventOrganizing-jedgedrudd" title="Event Organizing">📋</a> <a href="#fundingFinding-jedgedrudd" title="Funding Finding">🔍</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.rse.ox.ac.uk/"><img src="https://avatars3.githubusercontent.com/u/3770306?v=4?s=100" width="100px;" alt="Fergus Cooper"/><br /><sub><b>Fergus Cooper</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=fcooper8472" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=fcooper8472" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jonchapman1"><img src="https://avatars1.githubusercontent.com/u/28925818?v=4?s=100" width="100px;" alt="jonchapman1"/><br /><sub><b>jonchapman1</b></sub></a><br /><a href="#ideas-jonchapman1" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-jonchapman1" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/colinplease"><img src="https://avatars3.githubusercontent.com/u/44977104?v=4?s=100" width="100px;" alt="Colin Please"/><br /><sub><b>Colin Please</b></sub></a><br /><a href="#ideas-colinplease" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-colinplease" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cwmonroe"><img src="https://avatars.githubusercontent.com/u/92099819?v=4?s=100" width="100px;" alt="cwmonroe"/><br /><sub><b>cwmonroe</b></sub></a><br /><a href="#ideas-cwmonroe" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-cwmonroe" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gjo97"><img src="https://avatars.githubusercontent.com/u/18349157?v=4?s=100" width="100px;" alt="Greg"/><br /><sub><b>Greg</b></sub></a><br /><a href="#ideas-gjo97" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-gjo97" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://faraday.ac.uk"><img src="https://avatars2.githubusercontent.com/u/42166506?v=4?s=100" width="100px;" alt="Faraday Institution"/><br /><sub><b>Faraday Institution</b></sub></a><br /><a href="#financial-FaradayInstitution" title="Financial">💵</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bessman"><img src="https://avatars3.githubusercontent.com/u/1999462?v=4?s=100" width="100px;" alt="Alexander Bessman"/><br /><sub><b>Alexander Bessman</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Abessman" title="Bug reports">🐛</a> <a href="#example-bessman" title="Examples">💡</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dalbamont"><img src="https://avatars1.githubusercontent.com/u/19659095?v=4?s=100" width="100px;" alt="dalbamont"/><br /><sub><b>dalbamont</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=dalbamont" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/anandmy"><img src="https://avatars1.githubusercontent.com/u/34894671?v=4?s=100" width="100px;" alt="Anand Mohan Yadav"/><br /><sub><b>Anand Mohan Yadav</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=anandmy" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/weilongai"><img src="https://avatars1.githubusercontent.com/u/41424174?v=4?s=100" width="100px;" alt="WEILONG AI"/><br /><sub><b>WEILONG AI</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=weilongai" title="Code">💻</a> <a href="#example-weilongai" title="Examples">💡</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=weilongai" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lonnbornj"><img src="https://avatars2.githubusercontent.com/u/35983543?v=4?s=100" width="100px;" alt="lonnbornj"/><br /><sub><b>lonnbornj</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=lonnbornj" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=lonnbornj" title="Tests">⚠️</a> <a href="#example-lonnbornj" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/priyanshuone6"><img src="https://avatars.githubusercontent.com/u/64051212?v=4?s=100" width="100px;" alt="Priyanshu Agarwal"/><br /><sub><b>Priyanshu Agarwal</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=priyanshuone6" title="Tests">⚠️</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=priyanshuone6" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Apriyanshuone6" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Apriyanshuone6" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-priyanshuone6" title="Maintenance">🚧</a> <a href="#tutorial-priyanshuone6" title="Tutorials">✅</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DrSOKane"><img src="https://avatars.githubusercontent.com/u/42972513?v=4?s=100" width="100px;" alt="DrSOKane"/><br /><sub><b>DrSOKane</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=DrSOKane" title="Code">💻</a> <a href="#example-DrSOKane" title="Examples">💡</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=DrSOKane" title="Documentation">📖</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=DrSOKane" title="Tests">⚠️</a> <a href="#tutorial-DrSOKane" title="Tutorials">✅</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3ADrSOKane" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Saransh-cpp"><img src="https://avatars.githubusercontent.com/u/74055102?v=4?s=100" width="100px;" alt="Saransh Chopra"/><br /><sub><b>Saransh Chopra</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=Saransh-cpp" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Saransh-cpp" title="Tests">⚠️</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=Saransh-cpp" title="Documentation">📖</a> <a href="#tutorial-Saransh-cpp" title="Tutorials">✅</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3ASaransh-cpp" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-Saransh-cpp" title="Maintenance">🚧</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DavidMStraub"><img src="https://avatars.githubusercontent.com/u/10965193?v=4?s=100" width="100px;" alt="David Straub"/><br /><sub><b>David Straub</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3ADavidMStraub" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=DavidMStraub" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/maurosgroi"><img src="https://avatars.githubusercontent.com/u/37576773?v=4?s=100" width="100px;" alt="maurosgroi"/><br /><sub><b>maurosgroi</b></sub></a><br /><a href="#ideas-maurosgroi" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/asinghgaba"><img src="https://avatars.githubusercontent.com/u/77078706?v=4?s=100" width="100px;" alt="Amarjit Singh Gaba"/><br /><sub><b>Amarjit Singh Gaba</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=asinghgaba" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KennethNwanoro"><img src="https://avatars.githubusercontent.com/u/78538806?v=4?s=100" width="100px;" alt="KennethNwanoro"/><br /><sub><b>KennethNwanoro</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=KennethNwanoro" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=KennethNwanoro" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alibh95"><img src="https://avatars.githubusercontent.com/u/65511923?v=4?s=100" width="100px;" alt="Ali Hussain Umar Bhatti"/><br /><sub><b>Ali Hussain Umar Bhatti</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=alibh95" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=alibh95" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/molel-gt"><img src="https://avatars.githubusercontent.com/u/81125862?v=4?s=100" width="100px;" alt="Leshinka Molel"/><br /><sub><b>Leshinka Molel</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=molel-gt" title="Code">💻</a> <a href="#ideas-molel-gt" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tobykirk"><img src="https://avatars.githubusercontent.com/u/42966045?v=4?s=100" width="100px;" alt="tobykirk"/><br /><sub><b>tobykirk</b></sub></a><br /><a href="#ideas-tobykirk" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tobykirk" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tobykirk" title="Tests">⚠️</a> <a href="#tutorial-tobykirk" title="Tutorials">✅</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chuckliu1979"><img src="https://avatars.githubusercontent.com/u/13491954?v=4?s=100" width="100px;" alt="Chuck Liu"/><br /><sub><b>Chuck Liu</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Achuckliu1979" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=chuckliu1979" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/partben"><img src="https://avatars.githubusercontent.com/u/88316576?v=4?s=100" width="100px;" alt="partben"/><br /><sub><b>partben</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=partben" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://gavinw.me"><img src="https://avatars.githubusercontent.com/u/6828967?v=4?s=100" width="100px;" alt="Gavin Wiggins"/><br /><sub><b>Gavin Wiggins</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Awigging" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=wigging" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dion-w"><img src="https://avatars.githubusercontent.com/u/91852142?v=4?s=100" width="100px;" alt="Dion Wilde"/><br /><sub><b>Dion Wilde</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Adion-w" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=dion-w" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.ehtec.co"><img src="https://avatars.githubusercontent.com/u/48386220?v=4?s=100" width="100px;" alt="Elias Hohl"/><br /><sub><b>Elias Hohl</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=ehtec" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KAschad"><img src="https://avatars.githubusercontent.com/u/93784399?v=4?s=100" width="100px;" alt="KAschad"/><br /><sub><b>KAschad</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3AKAschad" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Vaibhav-Chopra-GT"><img src="https://avatars.githubusercontent.com/u/92637595?v=4?s=100" width="100px;" alt="Vaibhav-Chopra-GT"/><br /><sub><b>Vaibhav-Chopra-GT</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=Vaibhav-Chopra-GT" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bardsleypt"><img src="https://avatars.githubusercontent.com/u/54084289?v=4?s=100" width="100px;" alt="bardsleypt"/><br /><sub><b>bardsleypt</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Abardsleypt" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=bardsleypt" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ndrewwang"><img src="https://avatars.githubusercontent.com/u/56122552?v=4?s=100" width="100px;" alt="ndrewwang"/><br /><sub><b>ndrewwang</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Andrewwang" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=ndrewwang" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MichaPhilipp"><img src="https://avatars.githubusercontent.com/u/58085966?v=4?s=100" width="100px;" alt="MichaPhilipp"/><br /><sub><b>MichaPhilipp</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3AMichaPhilipp" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/abillscmu"><img src="https://avatars.githubusercontent.com/u/48105066?v=4?s=100" width="100px;" alt="Alec Bills"/><br /><sub><b>Alec Bills</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=abillscmu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/agriyakhetarpal"><img src="https://avatars.githubusercontent.com/u/74401230?v=4?s=100" width="100px;" alt="Agriya Khetarpal"/><br /><sub><b>Agriya Khetarpal</b></sub></a><br /><a href="#infra-agriyakhetarpal" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=agriyakhetarpal" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=agriyakhetarpal" title="Documentation">📖</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Aagriyakhetarpal" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/awadell1"><img src="https://avatars.githubusercontent.com/u/5857298?v=4?s=100" width="100px;" alt="Alex Wadell"/><br /><sub><b>Alex Wadell</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=awadell1" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=awadell1" title="Tests">⚠️</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=awadell1" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/iatzak"><img src="https://avatars.githubusercontent.com/u/112731474?v=4?s=100" width="100px;" alt="iatzak"/><br /><sub><b>iatzak</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=iatzak" title="Documentation">📖</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Aiatzak" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=iatzak" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ayeankit"><img src="https://avatars.githubusercontent.com/u/72691866?v=4?s=100" width="100px;" alt="Ankit Kumar"/><br /><sub><b>Ankit Kumar</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=ayeankit" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://aniketsinghrawat.vercel.app/"><img src="https://avatars.githubusercontent.com/u/31622972?v=4?s=100" width="100px;" alt="Aniket Singh Rawat"/><br /><sub><b>Aniket Singh Rawat</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=dikwickley" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=dikwickley" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jeromtom"><img src="https://avatars.githubusercontent.com/u/83979298?v=4?s=100" width="100px;" alt="Jerom Palimattom Tom"/><br /><sub><b>Jerom Palimattom Tom</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=jeromtom" title="Documentation">📖</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=jeromtom" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=jeromtom" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://bradyplanden.github.io"><img src="https://avatars.githubusercontent.com/u/55357039?v=4?s=100" width="100px;" alt="Brady Planden"/><br /><sub><b>Brady Planden</b></sub></a><br /><a href="#example-BradyPlanden" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.jsbrittain.com/"><img src="https://avatars.githubusercontent.com/u/98161205?v=4?s=100" width="100px;" alt="jsbrittain"/><br /><sub><b>jsbrittain</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=jsbrittain" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=jsbrittain" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arjxn-py"><img src="https://avatars.githubusercontent.com/u/104268427?v=4?s=100" width="100px;" alt="Arjun"/><br /><sub><b>Arjun</b></sub></a><br /><a href="#infra-arjxn-py" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=arjxn-py" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=arjxn-py" title="Documentation">📖</a> <a href="https://github.com/pybamm-team/PyBaMM/pulls?q=is%3Apr+reviewed-by%3Aarjxn-py" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chenzhao-py"><img src="https://avatars.githubusercontent.com/u/75906533?v=4?s=100" width="100px;" alt="CHEN ZHAO"/><br /><sub><b>CHEN ZHAO</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Achenzhao-py" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.aboutenergy.io/"><img src="https://avatars.githubusercontent.com/u/91731499?v=4?s=100" width="100px;" alt="darryl-ad"/><br /><sub><b>darryl-ad</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=darryl-ad" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Adarryl-ad" title="Bug reports">🐛</a> <a href="#ideas-darryl-ad" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/julian-evers"><img src="https://avatars.githubusercontent.com/u/133691040?v=4?s=100" width="100px;" alt="julian-evers"/><br /><sub><b>julian-evers</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=julian-evers" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://batterycontrolgroup.engin.umich.edu/"><img src="https://avatars.githubusercontent.com/u/633873?v=4?s=100" width="100px;" alt="Jason Siegel"/><br /><sub><b>Jason Siegel</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=js1tr3" title="Code">💻</a> <a href="#ideas-js1tr3" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tommaull"><img src="https://avatars.githubusercontent.com/u/101814207?v=4?s=100" width="100px;" alt="Tom Maull"/><br /><sub><b>Tom Maull</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=tommaull" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=tommaull" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ejfdickinson"><img src="https://avatars.githubusercontent.com/u/116663050?v=4?s=100" width="100px;" alt="ejfdickinson"/><br /><sub><b>ejfdickinson</b></sub></a><br /><a href="#ideas-ejfdickinson" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Aejfdickinson" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bobonice"><img src="https://avatars.githubusercontent.com/u/22030806?v=4?s=100" width="100px;" alt="bobonice"/><br /><sub><b>bobonice</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Abobonice" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=bobonice" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kratman"><img src="https://avatars.githubusercontent.com/u/10170302?v=4?s=100" width="100px;" alt="Eric G. Kratz"/><br /><sub><b>Eric G. Kratz</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=kratman" title="Documentation">📖</a> <a href="#infra-kratman" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Akratman" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=kratman" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=kratman" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://aitorres.com"><img src="https://avatars.githubusercontent.com/u/26191851?v=4?s=100" width="100px;" alt="Andrés Ignacio Torres"/><br /><sub><b>Andrés Ignacio Torres</b></sub></a><br /><a href="#infra-aitorres" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Agnik7"><img src="https://avatars.githubusercontent.com/u/77234005?v=4?s=100" width="100px;" alt="Agnik Bakshi"/><br /><sub><b>Agnik Bakshi</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=Agnik7" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RuiheLi"><img src="https://avatars.githubusercontent.com/u/84007676?v=4?s=100" width="100px;" alt="RuiheLi"/><br /><sub><b>RuiheLi</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=RuiheLi" title="Code">💻</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=RuiheLi" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chmabaur"><img src="https://avatars.githubusercontent.com/u/127507466?v=4?s=100" width="100px;" alt="chmabaur"/><br /><sub><b>chmabaur</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/issues?q=author%3Achmabaur" title="Bug reports">🐛</a> <a href="https://github.com/pybamm-team/PyBaMM/commits?author=chmabaur" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AbhishekChaudharii"><img src="https://avatars.githubusercontent.com/u/91185083?v=4?s=100" width="100px;" alt="Abhishek Chaudhari"/><br /><sub><b>Abhishek Chaudhari</b></sub></a><br /><a href="https://github.com/pybamm-team/PyBaMM/commits?author=AbhishekChaudharii" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shubhambhar007"><img src="https://avatars.githubusercontent.com/u/32607282?v=4?s=100" width="100px;" alt="Shubham Bhardwaj"/><br /><sub><b>Shubham Bhardwaj</b></sub></a><br /><a href="#infra-shubhambhar007" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jlauber18"><img src="https://avatars.githubusercontent.com/u/28939653?v=4?s=100" width="100px;" alt="Jonathan Lauber"/><br /><sub><b>Jonathan Lauber</b></sub></a><br /><a href="#infra-jlauber18" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/prady0t"><img src="https://avatars.githubusercontent.com/u/99216956?v=4?s=100" width="100px;" alt="Pradyot Ranjan"/><br /><sub><b>Pradyot Ranjan</b></sub></a><br /><a href="#infra-prady0t" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
