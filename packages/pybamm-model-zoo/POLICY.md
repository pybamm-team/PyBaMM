# PyBaMM model zoo policy

This policy sets out the terms on which models are contributed to, kept in, and removed from the PyBaMM model zoo (the "zoo"). By opening a pull request that adds or changes a model in the zoo, you agree to these terms.

In this policy:

- the **PyBaMM team** means the core members of the PyBaMM project;
- a **contributor** is anyone who submits a model to the zoo;
- a model's **maintainers** are the people listed under `maintainers` in its `model.toml` manifest and named for its folder in `.github/CODEOWNERS`.

## 1. Model tiers

Every model in the zoo belongs to one of two tiers, recorded in its manifest:

1. **Core models** are maintained by the PyBaMM team. They are tested in PyBaMM's merge gate, so a change to PyBaMM that breaks a core model cannot be merged until the model is fixed.
2. **Community models** are maintained by their contributors. They are tested on every pull request for information only, and a failure does not block changes to PyBaMM.

## 2. Maintenance

1. The PyBaMM team maintains core models only. It has no obligation to maintain, fix, update, or support any community model.
2. The maintainers of a community model are responsible for keeping it working with current releases of PyBaMM and of its dependencies, for reviewing changes to it, and for answering issues about it.
3. Changes to PyBaMM may break community models without notice. The status badge of each model reports whether it currently passes its tests.
4. If a community model breaks and its maintainers do not fix it, the PyBaMM team may accept a fix from anyone else in the community. Whoever submits that fix may be added as a maintainer of the model.

## 3. No endorsement and no warranty

1. Inclusion of a community model in the zoo does not mean the PyBaMM team endorses it. The PyBaMM team does not check, and does not vouch for, the correctness, accuracy, validity, or fitness for any purpose of community models, their parameters, or their results.
2. The automated checks the zoo runs show that a model is well defined and runs; they do not show that its physics is right.
3. All models in the zoo, core and community, are provided "as is", without warranty of any kind, as set out in the license (section 4). Users are responsible for judging whether a model is suitable for their use.

## 4. License

1. All models in the zoo are distributed under PyBaMM's [BSD 3-Clause license](https://github.com/pybamm-team/PyBaMM/blob/main/LICENSE.txt). By contributing a model, you agree to license it on those terms.
2. Contributors who want a model released under any other license must host it in their own repository. Such a repository can still be discovered by the zoo registry as an external model collection (see the [README](https://github.com/pybamm-team/PyBaMM/tree/main/packages/pybamm-model-zoo#external-model-collections)), and is then governed by its own license rather than by this policy.
3. Third-party packages that a model depends on keep their own licenses, which must be compatible with the BSD 3-Clause license.

## 5. Contributors' responsibilities

By contributing a model, you confirm that:

1. you have the right to contribute it under the terms in section 4, including any code, data, or parameter values it contains;
2. it does not knowingly infringe the copyright, patents, or other rights of anyone else, nor breach any confidentiality or contractual obligation;
3. it is properly cited, and gives credit to any prior work it is based on;
4. you, and the other maintainers you list, will meet the responsibilities in section 2.

All participation in the zoo is subject to PyBaMM's [Code of Conduct](https://github.com/pybamm-team/PyBaMM/blob/main/CODE-OF-CONDUCT.md).

## 6. Rejecting and removing models

The PyBaMM team decides which models are accepted into the zoo, and may decline a model or remove an existing one at its discretion. Reasons may include, but are not limited to, a model that:

1. is inappropriate, offensive, or in breach of the Code of Conduct;
2. infringes the rights of others, or whose license or provenance is unclear;
3. contains malicious, unsafe, or deliberately misleading code;
4. is known to be scientifically wrong and is not corrected;
5. has stayed broken, or has had no active maintainer, for an extended period;
6. duplicates a model already in the zoo or in PyBaMM without adding anything substantial;
7. places an unreasonable burden on the zoo's infrastructure, for example through excessive test times or dependencies;
8. falls outside the scope of PyBaMM.

Where practical, the PyBaMM team will give the maintainers notice and a chance to address the problem before removing a model. A removed model remains available in the repository's history.

## 7. Adoption into core

1. The PyBaMM team monitors community models and may, from time to time, adopt a community model as a core model. From then on it is maintained by the PyBaMM team.
2. The decision to adopt a model, or not, rests solely with the PyBaMM team. No contributor is entitled to have a model adopted, and the PyBaMM team need not give reasons for its decision.
3. On adoption, the PyBaMM team may change the model as it sees fit, including its code, interface, tests, and documentation, and may move it out of the zoo into PyBaMM itself. The original contributors keep credit for their work through the model's citation, which will be preserved.
4. The PyBaMM team may also return a core model to the community tier, or retire it, if it can no longer maintain it.

## 8. Changes to this policy

The PyBaMM team may change this policy at any time. Changes are made by pull request to this file, and apply to all models in the zoo, including those contributed before the change.
