#
# Native PyBaMM Meshes
#
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
            # Zero dimensional submesh case (only one point)
            if submesh_types[domain] == pybamm.SubMesh0D:
                submesh_pts[domain] = 1
            # other cases
            else:
                submesh_pts[domain] = {}
                if len(list(geometry[domain]["primary"].keys())) > 2:
                    raise pybamm.GeometryError
                for prim_sec in list(geometry[domain].keys()):
                    # skip over tabs key
                    if prim_sec != "tabs":
                        for var in list(geometry[domain][prim_sec].keys()):
                            # Raise error if the number of points for a particular
                            # variable haven't been provided, unless that variable
                            # doesn't appear in the geometry
                            if (
                                var.id not in var_id_pts.keys()
                                and var.domain[0] in geometry.keys()
                            ):
                                raise KeyError(
                                    """
                                    Points not given for a variable in domain {}
                                    """.format(
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
            for prim_sec_tabs, variables in geometry[domain].items():
                # process tab information if using 1 or 2D current collectors
                if prim_sec_tabs == "tabs":
                    for tab, position_size in variables.items():
                        for position_size, sym in position_size.items():
                            if isinstance(sym, pybamm.Symbol):
                                sym_eval = sym.evaluate()
                                geometry[domain][prim_sec_tabs][tab][
                                    position_size
                                ] = sym_eval
                else:
                    for spatial_variable, spatial_limits in variables.items():
                        for lim, sym in spatial_limits.items():
                            if isinstance(sym, pybamm.Symbol):
                                sym_eval = sym.evaluate()
                                geometry[domain][prim_sec_tabs][spatial_variable][
                                    lim
                                ] = sym_eval

        # Create submeshes
        for domain in geometry:
            if (
                domain == "current collector"
                and submesh_types[domain] != pybamm.SubMesh0D
            ):
                self[domain] = [
                    submesh_types[domain](
                        geometry[domain]["primary"],
                        submesh_pts[domain],
                        geometry[domain]["tabs"],
                    )
                ]
            else:
                if "secondary" in geometry[domain].keys():
                    repeats = 1
                    for var in geometry[domain]["secondary"].keys():
                        repeats *= submesh_pts[domain][var.id]  # note (specific to FV)
                else:
                    repeats = 1
                self[domain] = [
                    submesh_types[domain](
                        geometry[domain]["primary"], submesh_pts[domain]
                    )
                ] * repeats

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
        # Check that the final edge of each submesh is the same as the first edge of the
        # next submesh
        for i in range(len(submeshnames) - 1):
            for j in range(len(self[submeshnames[i]])):
                if (
                    self[submeshnames[i]][j].edges[-1]
                    != self[submeshnames[i + 1]][j].edges[0]
                ):
                    raise pybamm.DomainError("submesh edges are not aligned")

            coord_sys = self[submeshnames[i]][0].coord_sys
            coord_sys_r = self[submeshnames[i + 1]][0].coord_sys
            if coord_sys != coord_sys_r:
                raise pybamm.DomainError(
                    "trying to combine two meshes in different coordinate systems"
                )
        submeshes = [None] * len(self[submeshnames[0]])
        for i in range(len(self[submeshnames[0]])):
            combined_submesh_edges = np.concatenate(
                [self[submeshnames[0]][i].edges]
                + [self[submeshname][i].edges[1:] for submeshname in submeshnames[1:]]
            )
            coord_sys = self[submeshnames[0]][i].coord_sys
            submeshes[i] = pybamm.SubMesh1D(combined_submesh_edges, coord_sys)
        return submeshes

    def add_ghost_meshes(self):
        """
        Create meshes for potential ghost nodes on either side of each submesh, using
        self.submeshclass
        This will be useful for calculating the gradient with Dirichlet BCs.
        """
        # Get all submeshes relating to space (i.e. exclude time)
        submeshes = [
            (domain, submesh_list)
            for domain, submesh_list in self.items()
            if not isinstance(
                submesh_list[0], (pybamm.SubMesh0D, pybamm.ScikitSubMesh2D)
            )
        ]
        for domain, submesh_list in submeshes:

            self[domain + "_left ghost cell"] = [None] * len(submesh_list)
            self[domain + "_right ghost cell"] = [None] * len(submesh_list)
            for i, submesh in enumerate(submesh_list):
                edges = submesh.edges

                # left ghost cell: two edges, one node, to the left of existing submesh
                lgs_edges = np.array([2 * edges[0] - edges[1], edges[0]])
                self[domain + "_left ghost cell"][i] = pybamm.SubMesh1D(
                    lgs_edges, submesh.coord_sys
                )

                # right ghost cell: two edges, one node, to the right of
                # existing submesh
                rgs_edges = np.array([edges[-1], 2 * edges[-1] - edges[-2]])
                self[domain + "_right ghost cell"][i] = pybamm.SubMesh1D(
                    rgs_edges, submesh.coord_sys
                )


class MeshGenerator:
    """
    Base class for mesh generator objects that are used to generate submeshes
    that require input paramaters.
    """

    def __init__(self):
        pass

    def __call__(self):
        """
        Each Mesh Generator should implemented a call method which returns
        an instance of a submesh.
        """
        raise NotImplementedError
