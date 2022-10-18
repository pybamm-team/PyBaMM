#
# convert an expression tree into a pack model
#
import pybamm
from copy import deepcopy


class Pack(object):
    def __init__(self, built_model, num_cells):
        # this is going to be a work in progress for a while:
        # for now, will just do it at the julia level
        self.cell_size = built_model.size
        self.cell_model = built_model
        self._offset = self.cell_size
        self.built_model = built_model
        self._sv_done = []
        self._cells = (built_model,)
        self.repeat_cells(num_cells)

    def repeat_cells(self, num_cells):
        for n in range(num_cells - 1):
            self.add_new_cell()
            self._offset += self.cell_size
            self._sv_done = []
            print("adding cell {} of {}".format(n, num_cells))
        self.built_model = pybamm.NumpyConcatenation(*self._cells)
        self.built_model.set_id()
        print("done building pack")

    def add_new_cell(self):
        new_model = deepcopy(self.cell_model)
        # at some point need to figure out parameters
        self.add_offset_to_state_vectors(new_model)
        self._cells += (new_model,)

    def add_offset_to_state_vectors(self, symbol):
        # this function adds an offset to the state vectors
        new_y_slices = ()
        if isinstance(symbol, pybamm.StateVector):
            # need to make sure its in place
            if symbol.id not in self._sv_done:
                for this_slice in symbol.y_slices:
                    start = this_slice.start + self._offset
                    stop = this_slice.stop + self._offset
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


class InternalPackParameter(object):
    def __init__(self):
        pass
