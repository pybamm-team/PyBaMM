#
# Native PyBaMM Meshes
#
import numbers
import numpy as np
import pybamm


class Mesh(dict):
    """
    Mesh contains a list of submeshes on each subdomain.

    **Extends**: dict

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
        # convert var_pts to an id dict
        var_id_pts = {var.id: pts for var, pts in var_pts.items()}

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
                        # Raise error if the number of points for a particular
                        # variable haven't been provided, unless that variable
                        # doesn't appear in the geometry
                        if (
                            var.id not in var_id_pts.keys()
                            and var.domain[0] in geometry.keys()
                        ):
                            raise KeyError(
                                "Points not given for a variable in domain {}".format(
                                    domain
                                )
                            )
                        # Otherwise add to the dictionary of submesh points
                        submesh_pts[domain][var.id] = var_id_pts[var.id]
        self.submesh_pts = submesh_pts

        # Input domain order manually
        self.domain_order = []
        # First the macroscale domains, whose order we care about
        for domain in ["negative electrode", "separator", "positive electrode"]:
            if domain in geometry:
                self.domain_order.append(domain)
        # Then the remaining domains
        for domain in geometry:
            if domain not in ["negative electrode", "separator", "positive electrode"]:
                self.domain_order.append(domain)

        # evaluate any expressions in geometry
        for domain in geometry:
            for spatial_variable, spatial_limits in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if spatial_variable == "tabs":
                    for tab, position_size in spatial_limits.items():
                        for position_size, sym in position_size.items():
                            if isinstance(sym, pybamm.Symbol):
                                sym_eval = sym.evaluate()
                                geometry[domain]["tabs"][tab][position_size] = sym_eval
                else:
                    for lim, sym in spatial_limits.items():
                        if isinstance(sym, pybamm.Symbol):
                            sym_eval = sym.evaluate()
                        elif isinstance(sym, numbers.Number):
                            sym_eval = sym
                        geometry[domain][spatial_variable][lim] = sym_eval

        # Create submeshes
        for domain in geometry:
            self[domain] = submesh_types[domain](geometry[domain], submesh_pts[domain])

        # add ghost meshes
        self.add_ghost_meshes()

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
        # If there is just a single submesh, we can return it directly
        if len(submeshnames) == 1:
            return self[submeshnames[0]]
        # Check that the final edge of each submesh is the same as the first edge of the
        # next submesh
        for i in range(len(submeshnames) - 1):
            if self[submeshnames[i]].edges[-1] != self[submeshnames[i + 1]].edges[0]:
                raise pybamm.DomainError("submesh edges are not aligned")

            coord_sys = self[submeshnames[i]].coord_sys
            coord_sys_r = self[submeshnames[i + 1]].coord_sys
            if coord_sys != coord_sys_r:
                raise pybamm.DomainError(
                    "trying to combine two meshes in different coordinate systems"
                )
        combined_submesh_edges = np.concatenate(
            [self[submeshnames[0]].edges]
            + [self[submeshname].edges[1:] for submeshname in submeshnames[1:]]
        )
        coord_sys = self[submeshnames[0]].coord_sys
        submesh = pybamm.SubMesh1D(combined_submesh_edges, coord_sys)
        # add in internal boundaries
        submesh.internal_boundaries = [
            self[submeshname].edges[0] for submeshname in submeshnames[1:]
        ]

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
            if not isinstance(submesh, (pybamm.SubMesh0D, pybamm.ScikitSubMesh2D))
        ]
        for domain, submesh in submeshes:

            edges = submesh.edges

            # left ghost cell: two edges, one node, to the left of existing submesh
            lgs_edges = np.array([2 * edges[0] - edges[1], edges[0]])
            self[domain + "_left ghost cell"] = pybamm.SubMesh1D(
                lgs_edges, submesh.coord_sys
            )

            # right ghost cell: two edges, one node, to the right of
            # existing submesh
            rgs_edges = np.array([edges[-1], 2 * edges[-1] - edges[-2]])
            self[domain + "_right ghost cell"] = pybamm.SubMesh1D(
                rgs_edges, submesh.coord_sys
            )


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
        return "Generator for {}".format(self.submesh_type.__name__)
