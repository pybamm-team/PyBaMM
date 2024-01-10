(user_guide)=

# PyBaMM user guide

This guide is an overview and explains the important features;
details are found in [](api_docs).

```{toctree}
---
caption: Getting started
maxdepth: 1
---

installation/index
getting_started
```

```{toctree}
---
caption: Fundamentals and usage
maxdepth: 2
---
fundamentals/index
fundamentals/battery_models
```

```{toctree}
---
caption: Contributing guide
maxdepth: 1
---
contributing
```

# Example notebooks

PyBaMM ships with example notebooks that demonstrate how to use it and reveal some of its
functionalities and its inner workings. For more examples, see the [Examples](../examples/index.rst) section.

```{only} latex
The notebooks are not included in PDF formats of the documentation. You may access them on PyBaMM's hosted
documentation available at https://docs.pybamm.org/en/latest/source/examples/index.html
```

```{nbgallery}
---
caption: Getting Started
maxdepth: 1
glob:
---
../examples/notebooks/getting_started/tutorial-1-how-to-run-a-model.ipynb
../examples/notebooks/getting_started/tutorial-2-compare-models.ipynb
../examples/notebooks/getting_started/tutorial-3-basic-plotting.ipynb
../examples/notebooks/getting_started/tutorial-4-setting-parameter-values.ipynb
../examples/notebooks/getting_started/tutorial-5-run-experiments.ipynb
../examples/notebooks/getting_started/tutorial-6-managing-simulation-outputs.ipynb
../examples/notebooks/getting_started/tutorial-7-model-options.ipynb
../examples/notebooks/getting_started/tutorial-8-solver-options.ipynb
../examples/notebooks/getting_started/tutorial-9-changing-the-mesh.ipynb
../examples/notebooks/getting_started/tutorial-10-creating-a-model.ipynb
../examples/notebooks/getting_started/tutorial-11-creating-a-submodel.ipynb
```

```{nbgallery}
---
caption: Creating Models
maxdepth: 1
glob:
---
../examples/notebooks/creating_models/1-an-ode-model.ipynb
../examples/notebooks/creating_models/2-a-pde-model.ipynb
../examples/notebooks/creating_models/3-negative-particle-problem.ipynb
../examples/notebooks/creating_models/4-comparing-full-and-reduced-order-models.ipynb
../examples/notebooks/creating_models/5-half-cell-model.ipynb
../examples/notebooks/creating_models/6-a-simple-SEI-model.ipynb
```
