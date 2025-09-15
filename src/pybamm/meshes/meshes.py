#
# Native PyBaMM Meshes
#
import numbers
import warnings

import numpy as np

import pybamm


class Mesh(dict):
    """
    Mesh contains a list of submeshes on each subdomain.

    Parameters
    ----------

    geometry : :class: `Geometry`
        contains the geometry of the problem.
    submesh_types: dict
        contains the types of submeshes to use (e.g. Uniform1DSubMesh)
    submesh_pts: dict
        contains the number of points on each subdomain

    """

    def __init__(self, geometry, submesh_types, var_pts):
        super().__init__()

        # Save geometry
        self.geometry = geometry

        # Preprocess var_pts
        var_pts_input = var_pts
        var_pts = {}
        for key, value in var_pts_input.items():
            if isinstance(key, str):
                key = getattr(pybamm.standard_spatial_vars, key)
            var_pts[key] = value

        # convert var_pts to an id dict
        var_name_pts = {var.name: pts for var, pts in var_pts.items()}

        for domain, generator in submesh_types.items():
            if isinstance(generator, pybamm.ScikitFemGenerator3D):
                for var in geometry[domain]:
                    if isinstance(var, str):
                        var_name = var
                    else:
                        var_name = var.name
                    provided_pts = var_name_pts.get(var_name)
                    if provided_pts is not None:
                        pybamm.logger.warning(
                            f"For the 3D FEM submesh on domain '{domain}', the value "
                            f"of all 'var_pts' are ignored. "
                            "The mesh is generated to satisfy the 'h' parameter. "
                            "It is recommended to pass 'None' for spatial variables "
                            "on a 3D FEM domain to avoid this warning."
                        )
                        # We only need to warn once per domain.
                        break

        # create submesh_pts from var_pts
        submesh_pts = {}
        for domain in geometry:
            # create mesh generator if just class is passed (will throw an error
            # later if the mesh needed parameters)
            if not isinstance(
                submesh_types[domain], pybamm.MeshGenerator
            ) and issubclass(submesh_types[domain], pybamm.SubMesh):
                submesh_types[domain] = pybamm.MeshGenerator(submesh_types[domain])
            # Zero dimensional submesh case (only one point)
            if issubclass(submesh_types[domain].submesh_type, pybamm.SubMesh0D):
                submesh_pts[domain] = 1
            # other cases
            else:
                submesh_pts[domain] = {}
                if len(list(geometry[domain].keys())) > 3:
                    raise pybamm.GeometryError("Too many keys provided")
                for var in list(geometry[domain].keys()):
                    if var in ["primary", "secondary"]:
                        raise pybamm.GeometryError(
                            "Geometry should no longer be given keys 'primary' or "
                            "'secondary'. See pybamm.battery_geometry() for example"
                        )
                    # skip over tabs key
                    if var != "tabs":
                        if isinstance(var, str):
                            var = getattr(pybamm.standard_spatial_vars, var)
                        # Raise error if the number of points for a particular
                        # variable haven't been provided, unless that variable
                        # doesn't appear in the geometry
                        if (
                            var.name not in var_name_pts.keys()
                            and var.domain[0] in geometry.keys()
                        ):
                            raise KeyError(
                                f"Points not given for variable '{var.name}' in domain '{domain}'"
                            )
                        # Otherwise add to the dictionary of submesh points
                        submesh_pts[domain][var.name] = var_name_pts[var.name]
        self.submesh_pts = submesh_pts

        # evaluate any expressions in geometry
        for domain in geometry:
            for spatial_variable, spatial_limits in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if spatial_variable == "tabs":
                    for tab, position_info in spatial_limits.items():
                        for position_size, sym in position_info.items():
                            if isinstance(sym, pybamm.Symbol):
                                sym_eval = sym.evaluate()
                                geometry[domain]["tabs"][tab][position_size] = sym_eval
                else:
                    for lim, sym in spatial_limits.items():
                        if isinstance(sym, pybamm.Symbol):
                            try:
                                sym_eval = sym.evaluate()
                            except KeyError:
                                sym_eval = sym
                            except NotImplementedError as error:
                                if sym.has_symbol_of_classes(pybamm.Parameter):
                                    raise pybamm.DiscretisationError(
                                        "Parameter values have not yet been set for "
                                        "geometry. Make sure that something like "
                                        "`param.process_geometry(geometry)` has been "
                                        "run."
                                    ) from error
                                else:
                                    raise error
                        elif isinstance(sym, numbers.Number):
                            sym_eval = sym
                        geometry[domain][spatial_variable][lim] = sym_eval

        # Create submeshes
        self.base_domains = []
        for domain in geometry:
            self[domain] = submesh_types[domain](geometry[domain], submesh_pts[domain])
            self.base_domains.append(domain)

        # add ghost meshes
        self.add_ghost_meshes()

    @classmethod
    def _from_json(cls, snippet: dict):
        instance = cls.__new__(cls)
        super(Mesh, instance).__init__()

        instance.submesh_pts = snippet["submesh_pts"]
        instance.base_domains = snippet["base_domains"]

        for k, v in snippet["sub_meshes"].items():
            instance[k] = v

        # instance.add_ghost_meshes()

        return instance

    def __getitem__(self, domains):
        if isinstance(domains, str):
            domains = (domains,)
        domains = tuple(domains)
        try:
            return super().__getitem__(domains)
        except KeyError:
            value = self.combine_submeshes(*domains)
            self[domains] = value
            return value

    def __setitem__(self, domains, value):
        if isinstance(domains, str):
            domains = (domains,)
        super().__setitem__(domains, value)

    def combine_submeshes(self, *submeshnames):
        """Combine submeshes into a new submesh, using self.submeshclass
        Raises pybamm.DomainError if submeshes to be combined do not match up (edges are
        not aligned).

        Parameters
        ----------
        submeshnames: list of str
            The names of the submeshes to be combined

        Returns
        -------
        submesh: :class:`self.submeshclass`
            A new submesh with the class defined by self.submeshclass
        """
        if submeshnames == ():
            raise ValueError("Submesh domains being combined cannot be empty")
        # Check that the final edge of each submesh is the same as the first edge of the
        # next submesh
        # TODO: We need a more robust way to check whether the submeshes are being combined lr or tb
        for i in range(len(submeshnames) - 1):
            if self[submeshnames[i]].dimension != self[submeshnames[i + 1]].dimension:
                raise pybamm.GeometryError(
                    "Cannot combine submeshes of different dimensions"
                )
            elif self[submeshnames[i]].dimension == 2:
                if "left" in submeshnames[i] or "right" in submeshnames[i + 1]:
                    # Make sure that the lr edges are aligned
                    if (
                        self[submeshnames[i]].edges_lr[-1]
                        != self[submeshnames[i + 1]].edges_lr[0]
                    ):
                        raise pybamm.DomainError("lr edges are not aligned")
                    elif (
                        self[submeshnames[i]].edges_tb
                        != self[submeshnames[i + 1]].edges_tb
                    ).any():
                        raise pybamm.DomainError("tb edges are not aligned")
                    else:
                        pass

                elif "top" in submeshnames[i] or "bottom" in submeshnames[i + 1]:
                    # Make sure that the tb edges are aligned
                    if (
                        self[submeshnames[i]].edges_tb[-1]
                        != self[submeshnames[i + 1]].edges_tb[0]
                    ):
                        raise pybamm.DomainError("tb edges are not aligned")
                    elif (
                        self[submeshnames[i]].edges_lr
                        != self[submeshnames[i + 1]].edges_lr
                    ).any():
                        raise pybamm.DomainError("lr edges are not aligned")
                    else:
                        pass
                pass
            elif self[submeshnames[i]].edges[-1] == self[submeshnames[i + 1]].edges[0]:
                # submeshes are aligned, all good
                pass
            elif hasattr(self[submeshnames[i]], "min") or hasattr(
                self[submeshnames[i + 1]], "min"
            ):
                # we have to give benefit of the doubt if either is symbolic because we won't know length until we have processed parameters.
                pass
            else:
                # submeshes are not aligned and neither is symbolic
                raise pybamm.DomainError("submesh edges are not aligned")

            coord_sys = self[submeshnames[i]].coord_sys
            coord_sys_r = self[submeshnames[i + 1]].coord_sys
            if coord_sys != coord_sys_r:
                raise pybamm.DomainError(
                    "trying to combine two meshes in different coordinate systems"
                )

        coord_sys = self[submeshnames[0]].coord_sys
        if self[submeshnames[0]].dimension == 1:
            combined_submesh_edges = np.concatenate(
                [self[submeshnames[0]].edges]
                + [self[submeshname].edges[1:] for submeshname in submeshnames[1:]]
            )
            submesh = pybamm.SubMesh1D(combined_submesh_edges, coord_sys)
        elif self[submeshnames[0]].dimension == 2:
            # If it's an lr concatenation, then we only need to concatenate the edges_lr
            if "left" in submeshnames[0] or "right" in submeshnames[-1]:
                combined_submesh_edges_lr = np.concatenate(
                    [self[submeshnames[0]].edges_lr]
                    + [
                        self[submeshname].edges_lr[1:]
                        for submeshname in submeshnames[1:]
                    ]
                )
                combined_submesh_edges_tb = self[submeshnames[0]].edges_tb
            elif "top" in submeshnames[0] or "bottom" in submeshnames[-1]:
                combined_submesh_edges_tb = np.concatenate(
                    [self[submeshnames[0]].edges_tb]
                    + [
                        self[submeshname].edges_tb[1:]
                        for submeshname in submeshnames[1:]
                    ]
                )
                combined_submesh_edges_lr = self[submeshnames[0]].edges_lr
            else:
                warnings.warn(
                    "Could not determine how to combine submeshes. Assuming left-right concatenation.",
                    stacklevel=2,
                )
                combined_submesh_edges_lr = np.concatenate(
                    [self[submeshnames[0]].edges_lr]
                    + [
                        self[submeshname].edges_lr[1:]
                        for submeshname in submeshnames[1:]
                    ]
                )
                combined_submesh_edges_tb = self[submeshnames[0]].edges_tb
            submesh = pybamm.SubMesh2D(
                combined_submesh_edges_lr, combined_submesh_edges_tb, coord_sys
            )

        if getattr(self[submeshnames[0]], "length", None) is not None:
            # Assume that the ghost cells have the same length as the first submesh
            if any("ghost" in submeshname for submeshname in submeshnames):
                submesh_min = self[submeshnames[0]].min
                submesh_length = self[submeshnames[0]].length
            # If not ghost cells, then the length is the sum of the lengths of the submeshes
            else:
                submesh_min = self[submeshnames[0]].min
                submesh_length = sum(
                    [self[submeshname].length for submeshname in submeshnames]
                )
            submesh.length = submesh_length
            submesh.min = submesh_min
        # add in internal boundaries
        for i, submeshname in enumerate(submeshnames[1:]):
            i = i + 1
            if getattr(self[submeshname], "length", None) is not None:
                min = self[submeshname].min
            else:
                min = 0
            if submesh.dimension == 1:
                submesh.internal_boundaries.append(self[submeshname].edges[0] + min)
            elif (
                "left" in submeshname
                or "left" in submeshnames[i - 1]
                or "right" in submeshname
            ):
                submesh.internal_boundaries.append(self[submeshname].edges_lr[0] + min)
            elif (
                "top" in submeshname
                or "top" in submeshnames[i - 1]
                or "bottom" in submeshname
            ):
                submesh.internal_boundaries.append(self[submeshname].edges_tb[0] + min)
            else:
                warnings.warn(
                    "Could not determine how to combine submeshes. Assuming left-right concatenation.",
                    stacklevel=2,
                )
                submesh.internal_boundaries.append(self[submeshname].edges_lr[0] + min)
        return submesh

    def add_ghost_meshes(self):
        """
        Create meshes for potential ghost nodes on either side of each submesh, using
        self.submeshclass
        This will be useful for calculating the gradient with Dirichlet BCs.
        """
        # Get all submeshes relating to space (i.e. exclude time)
        submeshes = [
            (domain, submesh)
            for domain, submesh in self.items()
            if (
                len(domain) == 1
                and not isinstance(
                    submesh,
                    pybamm.SubMesh0D
                    | pybamm.ScikitSubMesh2D
                    | pybamm.ScikitFemSubMesh3D,
                )
            )
        ]
        for domain, submesh in submeshes:
            if submesh.dimension == 1:
                self[domain[0] + "_left ghost cell"] = submesh.create_ghost_cell("left")
                self[domain[0] + "_right ghost cell"] = submesh.create_ghost_cell(
                    "right"
                )
            elif submesh.dimension == 2:
                self[domain[0] + "_left ghost cell"] = submesh.create_ghost_cell("left")
                self[domain[0] + "_right ghost cell"] = submesh.create_ghost_cell(
                    "right"
                )
                self[domain[0] + "_bottom ghost cell"] = submesh.create_ghost_cell(
                    "bottom"
                )
                self[domain[0] + "_top ghost cell"] = submesh.create_ghost_cell("top")
            else:
                raise NotImplementedError(
                    "ghost cells not implemented for submeshes of dimension "
                    + str(submesh.dimension)
                )

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry

    def to_json(self):
        json_dict = {
            "submesh_pts": self.submesh_pts,
            "base_domains": self.base_domains,
        }

        return json_dict


class SubMesh:
    """
    Base submesh class.
    Contains the position of the nodes, the number of mesh points, and
    (optionally) information about the tab locations.
    """

    def __init__(self):
        pass


class MeshGenerator:
    """
    Base class for mesh generator objects that are used to generate submeshes.

    Parameters
    ----------

    submesh_type: :class:`pybamm.SubMesh`
        The type of submesh to use (e.g. Uniform1DSubMesh).
    submesh_params: dict, optional
        Contains any parameters required by the submesh.
    """

    def __init__(self, submesh_type, submesh_params=None):
        self.submesh_type = submesh_type
        self.submesh_params = submesh_params or {}

    def __call__(self, lims, npts):
        return self.submesh_type(lims, npts, **self.submesh_params)

    def __repr__(self):
        return f"Generator for {self.submesh_type.__name__}"
