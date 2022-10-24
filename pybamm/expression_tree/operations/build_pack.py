#
# convert an expression tree into a pack model
#

#TODO
# - Build Batteries
# -- Current In Batteries
# -- Terminal voltage from batteries
# - Test sign convention
# - Eliminate node1x & node1y (use graph only)
# - Thermals
import pybamm
from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
import liionpack as lp


class Pack(object):
    def __init__(self, model, netlist, parameter_values=None):
        # this is going to be a work in progress for a while:
        # for now, will just do it at the julia level

        # Build the cell expression tree with necessary parameters.
        # think about moving this to a separate function.
        if parameter_values is not None:
            raise AssertionError("parameter values not supported")
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {"Current function [A]": pybamm.PsuedoInputParameter("cell_current")}
        )
        self.cell_parameter_values = parameter_values

        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sim.build()
        self.cell_model = pybamm.numpy_concatenation(
            sim.built_model.concatenated_rhs, sim.built_model.concatenated_algebraic
        )
        self.cell_size = self.cell_model.shape[0]
        self._sv_done = []

        self.netlist = netlist
        self.cell_currents = {}

        self.process_netlist()

        # get x and y coords for nodes from graph.
        node_xs = [n for n in range(max(self.circuit_graph.nodes) + 1)]
        node_ys = [n for n in range(max(self.circuit_graph.nodes) + 1)]
        for row in netlist.itertuples():
            node_xs[row.node1] = row.node1_x
            node_ys[row.node1] = row.node1_y
        self.node_xs = node_xs
        self.node_ys = node_ys

    def process_netlist(self):
        curr = [{} for i in range(len(self.netlist))]
        self.netlist.insert(0, "currents", curr)

        self.netlist = self.netlist.rename(
            columns={"node1": "source", "node2": "target"}
        )
        self.netlist["positive_node"] = self.netlist["source"]
        self.netlist["negative_node"] = self.netlist["target"]
        self.circuit_graph = nx.from_pandas_edgelist(self.netlist, edge_attr=True)

    #Function that adds new cells, and puts them in the appropriate places.
    def add_new_cell(self):
        # TODO: deal with variables dict here too.
        # This is the place to get clever.
        new_model = deepcopy(self.cell_model)
        # at some point need to figure out parameters
        self.add_offset_to_state_vectors(new_model)
        new_model.set_id()
        return new_model

    def add_offset_to_state_vectors(self, symbol):
        # this function adds an offset to the state vectors
        new_y_slices = ()
        if isinstance(symbol, pybamm.StateVector):
            # need to make sure its in place
            if symbol.id not in self._sv_done:
                for this_slice in symbol.y_slices:
                    start = this_slice.start + self.offset
                    stop = this_slice.stop + self.offset
                    step = this_slice.step
                    new_slice = slice(start, stop, step)
                    new_y_slices += (new_slice,)
                symbol.replace_y_slices(*new_y_slices)
                symbol.set_id()
                self._sv_done += [symbol.id]

        elif isinstance(symbol, pybamm.StateVectorDot):
            raise NotImplementedError("Idk what this means")
        else:
            for child in symbol.children:
                self.add_offset_to_state_vectors(child)
                child.set_id()
            symbol.set_id()

    def build_pack(self):
        # this function builds expression trees to compute the current.

        # cycle basis is the list of loops over which we will do kirchoff mesh analysis
        mcb = nx.minimum_cycle_basis(self.circuit_graph)

        # generate loop currents and current source voltages-- this is what we don't know.
        num_loops = len(mcb)

        curr_sources = [edge for edge in self.circuit_graph.edges if self.circuit_graph.edges[edge]["desc"][0]=="I"]
        num_curr_sources = len(curr_sources)

        loop_currents = [
            pybamm.StateVector(slice(n, n + 1), name="current_{}".format(n))
            for n in range(num_loops)
        ]

        curr_sources = []
        n = num_loops
        for edge in self.circuit_graph.edges:
            if self.circuit_graph.edges[edge]["desc"][0] == "I":
                self.circuit_graph.edges[edge]["voltage"] = pybamm.StateVector(slice(n, n + 1), name="current_source_{}".format(n))
                n += 1
                curr_sources.append(edge)

        # now we know the offset, we should "build" the batteries here. will still need to replace the currents later.
        self.offset = len(loop_currents) + len(curr_sources)
        self.batteries = {}
        for desc in self.netlist.desc:
            if desc[0] == "V":
                new_cell = self.add_new_cell()
                self.batteries.update({desc: new_cell})
                self.offset += self.cell_size

        if len(curr_sources) != 1:
            raise NotImplementedError("can't do this yet")
        # copy the basis which we can use to place the loop currents
        basis_to_place = deepcopy(mcb)

        self.place_currents(loop_currents, basis_to_place)

        pack_eqs = self.build_pack_equations(loop_currents, curr_sources)






    def build_pack_equations(self, loop_currents, curr_sources):
        # start by looping through the loop currents. Sum Voltages
        pack_equations = []
        for i, loop_current in enumerate(loop_currents):
            # loop through the edges
            eq = []
            for edge in self.circuit_graph.edges:
                if loop_current in self.circuit_graph.edges[edge]["currents"]:
                    #get the name of the edge current. 
                    edge_type = self.circuit_graph.edges[edge]["desc"][0]
                    direction = self.circuit_graph.edges[edge]["currents"][loop_current]
                    this_edge_current = loop_current
                    for current in self.circuit_graph.edges[edge]["currents"]:
                        if current == loop_current:
                            continue
                        if self.circuit_graph.edges[edge]["currents"][current] == "positive":
                            this_edge_current = this_edge_current + current
                        else:
                            this_edge_current = this_edge_current - current
                    if edge_type == "R":
                        eq.append(this_edge_current * self.circuit_graph.edges[edge]["value"])
                    elif edge_type == "I":
                        curr_source_num = self.circuit_graph.edges[edge]["desc"][1:]
                        if curr_source_num != "0":
                            raise NotImplementedError(
                                "multiple current sources is not yet supported"
                            )

                        if direction == "positive":
                            eq.append(self.circuit_graph.edges[edge]["voltage"])
                        else:
                            eq.append(-self.circuit_graph.edges[edge]["voltage"])
                    elif edge_type == "V":
                        #
                        voltage = self.batteries[self.circuit_graph.edges[edge]["desc"]]
                        if direction =="positive":
                            eq.append(voltage)
                        else:
                            eq.append(-voltage)

            if len(eq) == 0:
                raise NotImplementedError(
                    "packs must include at least 1 circuit element"
                )
            elif len(eq) == 1:
                expr = eq[0]
            else:
                expr = eq[0] + eq[1]
                for e in range(2, len(eq)):
                    expr = expr + eq[e]
            # add equation to the pack.
            pack_equations.append(expr)

        # then loop through the current source voltages. Sum Currents.
        for i,curr_source in enumerate(curr_sources):
            currents = list(self.circuit_graph.edges[curr_source]["currents"])
            if self.circuit_graph.edges[curr_source]["currents"][currents[0]] == "positive":
                expr = currents[0]
            else:
                expr = -currents[0]
            for current in currents[1:]:
                if self.circuit_graph.edges[curr_source]["currents"][current] == "positive":
                    expr = expr+current
                else:
                    expr = expr-current
            pack_equations.append(expr)
        
        #concatenate all the pack equations and return it.
        pack_eqs = pybamm.numpy_concatenation(*pack_equations)
        return pack_eqs


            

    # This function places the currents on the edges in a predefined order.
    # it begins at loop 0, and loops through each "loop" -- really a cycle
    # of the mcb (minimum cycle basis) of the graph which defines the circuit.
    # Once it finds a loop in which the current node is in, it places the
    # loop current on each edge. Once the loop is finished, it removes the
    # loop and then proceeds to the next node and does the same thing. It
    # loops until all the loop currents have been placed.
    def place_currents(self, loop_currents, mcb):
        bottom_loop = 0
        for this_loop, loop in enumerate(mcb):
            for node in sorted(self.circuit_graph.nodes):
                if node in loop:
                    # setting var to remove the loop later
                    done_nodes = set()
                    # doesn't actually matter where we start.
                    # loop will always be a set.
                    if len(loop) != len(set(loop)):
                        raise NotImplementedError()
                    inner_node = node
                    # calculate the centroid of the loop
                    loop_xs = [self.node_xs[n] for n in loop]
                    loop_ys = [self.node_ys[n] for n in loop]
                    centroid_x = np.mean(loop_xs)
                    centroid_y = np.mean(loop_ys)
                    last_one = False
                    while True:
                        done_nodes.add(inner_node)

                        my_neighbors = set(
                            self.circuit_graph.neighbors(inner_node)
                        ).intersection(set(loop))

                        # if there are no neighbors in the group that have not been done, ur done!
                        my_neighbors = my_neighbors - done_nodes

                        if len(my_neighbors) == 0:
                            break
                        elif len(loop) == len(done_nodes) + 1 and not last_one:
                            last_one = True
                            done_nodes.remove(node)

                        # calculate the angle to all the neighbors.
                        # then, to go clockwise, pick the one with
                        # the largest angle.
                        my_x = self.node_xs[inner_node]
                        my_y = self.node_ys[inner_node]
                        angles = {
                            n: np.arctan2(
                                self.node_xs[n] - centroid_x,
                                self.node_ys[n] - centroid_y,
                            )
                            for n in my_neighbors
                        }
                        next_node = max(angles, key=angles.get)
                        # print("at node {}, now going to node {}".format(inner_node, next_node))
                        # print(len(angles))

                        # now, define the vector from the current node to the next node.
                        next_coords = [
                            self.node_xs[next_node] - my_x,
                            self.node_ys[next_node] - my_y,
                        ]

                        # go find the edge.

                        edge = self.circuit_graph.edges.get((inner_node, next_node))
                        if edge is None:
                            edge = self.circuit_graph.edges.get((next_node, inner_node))
                        if edge is None:
                            raise KeyError("uh oh")

                        # add this current to the loop.
                        if inner_node == edge["positive_node"]:
                            direction = "negative"
                        else:
                            direction = "positive"

                        edge["currents"].update(
                            {loop_currents[this_loop]: direction}
                        )
                        inner_node = next_node
                    break
