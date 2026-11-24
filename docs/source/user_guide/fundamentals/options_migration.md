(options_migration)=

# Battery model options in 26.11

In 26.11, battery model options are resolved and checked per electrode and per particle
phase. Most of the changes need no action from you:

- Options that are now rejected raise an `OptionError` that names the option, the
  electrode or phase, and what to change.
- Options that PyBaMM sets from other options raise a
  `pybamm.OptionDefaultDeprecationWarning` naming them. Pass those options with the
  values shown to keep the current behaviour.
- Models saved with an earlier version load with that version's option checks.

This page lists the changes that give a different model from the same options.

## Options that now give a different model

| Options you pass | Option that changes | Before 26.11 | From 26.11 | Warning |
|---|---|---|---|---|
| `"particle phases": ("1", "1")` | `"surface form"` | `"algebraic"` | `"false"` | None |
| `"loss of active material": ("stress-driven", "none")` | `"particle mechanics"` | `"swelling only"` | `("swelling only", "none")` | Yes |
| | `"stress-induced diffusion"` | `"true"` | `("true", "false")` | Yes |
| `"particle mechanics": ("swelling only", "none")` | `"stress-induced diffusion"` | `"true"` | `("true", "false")` | Yes |
| `"lithium plating": ("none", "partially reversible")` | `"SEI"` | `"none"` | `("none", "constant")` | Yes |

Before 26.11, these defaults either applied a value set on one electrode to both, or
ignored the positive electrode. They now follow each electrode. With `("1", "1")`, the
electrodes are treated as single phase, as with the default `"1"`.

The deprecation warning shows the new value, not the old one. To keep the old model,
pass the old values explicitly. For `("1", "1")`, pass `"surface form": "algebraic"`. For
the loss of active material case, pass:

```python
model = pybamm.lithium_ion.DFN(
    {
        "loss of active material": ("stress-driven", "none"),
        "particle mechanics": "swelling only",
        "stress-induced diffusion": "true",
    }
)
```

The old values in the last two rows are now rejected with an `OptionError`, so they can't
be restored. Before 26.11, the mechanics case applied stress-induced diffusion to an
electrode with no mechanics model. The plating case had no SEI on that electrode, which
fails during parameter processing when dead lithium decays through the SEI ([#5709](https://github.com/pybamm-team/PyBaMM/issues/5709)).

## Fixes that can change results

- `Yang2017` keeps the other options you give it. Before 26.11 it replaced them with its
  own, so `Yang2017({"thermal": "lumped"})` was isothermal.
- If only one electrode has a particle-size distribution, that electrode now gets the
  size-distribution stress and crack equations.
- In models saved before 26.11, a per-phase `"SEI"` value, such as
  `("none", "reaction limited")` on one electrode, now applies to each phase. This
  changes results only with `"SEI porosity change"` or `"SEI on cracks"`.

## Reading options

- `model.options["SEI"]`, `model.options["SEI on cracks"]` and
  `model.options["lithium plating"]` return the value you passed. A scalar such as
  `"reaction limited"` used to be returned as `("reaction limited", "none")`. Use
  `model.options.negative["SEI"]` and `model.options.positive["SEI"]` to read the
  value for each electrode.
- Model constructors and `pybamm.BatteryModelOptions` no longer change the options
  dictionary you pass in. Read the resolved values from `model.options`.
