#
# Compile a PyBaMM expression tree to Julia Code
#
import pybamm
import numpy as np
import scipy
from collections import OrderedDict
from math import floor
import graphlib

class FunctionRepeat(object):
    def __init__(self, expr, inputs):
        pass
    pass


def remove_lines_with(input_string, pattern):
    string_list = input_string.split("\n")
    my_string = ""
    for s in string_list:
        if pattern not in s:
            my_string = my_string + s + "\n"
    return my_string

#Wrapper to designate a function.
#Bottom needs to already have a julia
#conversion. (for shape.)
class PybammJuliaFunction(pybamm.Symbol):
    def __init__(self, children, expr, name):
        self.expr = expr
        super().__init__(name, children)
    def evaluate_for_shape(self):
        return self.expr.evaluate_for_shape()
    
    @property
    def shape(self):
        return self.expr.shape
    




class JuliaConverter(object):
    def __init__(
        self,
        ismtk=False,
        cache_type="standard",
        jacobian_type="analytical",
        preallocate=True,
        dae_type="semi-explicit",
        input_parameter_order=None,
        inline=True,
        parallel="legacy-serial",
        outputs = [],
        inputs = [],
        black_box = False
    ):
        #if len(outputs) != 1:
        #    raise NotImplementedError("Julia black box can only have 1 output")
        
        if ismtk:
            raise NotImplementedError("mtk is not supported")

        if input_parameter_order is None:
            input_parameter_order = []

        if parallel != "legacy-serial" and inline:
            raise NotImplementedError(
                "Inline not supported with anything other than legacy-serial"
            )

        # Characteristics
        self._cache_type = cache_type
        self._ismtk = ismtk
        self._jacobian_type = jacobian_type
        self._preallocate = preallocate
        self._dae_type = dae_type

        self._type = "Float64"
        self._inline = inline
        self._parallel = parallel
        self._black_box = black_box
        # "Caches"
        # Stores Constants to be Declared in the initial cache
        # insight: everything is just a line of code

        # INTERMEDIATE: A List of Stuff to do.
        # Keys are ID's and lists are variable names.
        self._intermediate = OrderedDict()

        # Cache Dict and Const Dict Host Julia
        # Variable Names to be used to generate the code.
        self._cache_dict = OrderedDict()
        self._const_dict = OrderedDict()

        # the real hero
        self._dag = {}
        self._code = {}

        self.input_parameter_order = input_parameter_order

        self._cache_id = 0
        self._const_id = 0

        self._cache_and_const_string = ""
        self.function_definition = ""
        self._function_string = ""
        self._return_string = ""
        self._cache_initialization_string = ""

    def cache_exists(self, my_id, inputs):
        existance = self._cache_dict.get(my_id) is not None
        if existance:
            for this_input in inputs:
                self._dag[my_id].add(this_input)
        return self._cache_dict.get(my_id) is not None

    # know where to go to find a variable.
    # this could be smoother, there will
    # need to be a ton of boilerplate here.
    # This function breaks down and analyzes
    # any binary tree. Will fail if used on
    # a non-binary tree.
    def break_down_binary(self, symbol):
        # Check for constant
        # assert not is_constant_and_can_evaluate(symbol)
        # take care of the kids first (this is recursive
        # but multiple-dispatch recursive which is cool)
        id_left = self._convert_tree_to_intermediate(symbol.children[0])
        id_right = self._convert_tree_to_intermediate(symbol.children[1])
        my_id = symbol.id
        return id_left, id_right, my_id

    def break_down_concatenation(self, symbol):
        child_ids = []
        for child in symbol.children:
            child_id = self._convert_tree_to_intermediate(child)
            child_ids.append(child_id)
        first_id = child_ids[0]
        num_cols = self._intermediate[first_id].shape[1]
        num_rows = 0
        for child_id in child_ids:
            child_shape = self._intermediate[child_id].shape
            num_rows += child_shape[0]
        shape = (num_rows, num_cols)
        return child_ids, shape

    # Convert-Trees go here

    # Binary trees constructors. All follow the pattern of mat-mul.
    # They need to find their shapes, assuming that the shapes of
    # the nodes one level below them in the expression tree have
    # already been computed.
    def _convert_tree_to_intermediate(self, symbol):
        if isinstance(symbol, pybamm.NumpyConcatenation):
            my_id = symbol.id
            children_julia, shape = self.break_down_concatenation(symbol)
            self._intermediate[my_id] = JuliaNumpyConcatenation(
                my_id, shape, children_julia
            )
        elif isinstance(symbol, pybamm.SparseStack):
            my_id = symbol.id
            children_julia, shape = self.break_down_concatenation(symbol)
            self._intermediate[my_id] = JuliaSparseStack(my_id, shape, children_julia)
        elif isinstance(symbol, pybamm.DomainConcatenation):
            my_id = symbol.id
            children_julia, shape = self.break_down_concatenation(symbol)
            self._intermediate[my_id] = JuliaDomainConcatenation(
                my_id,
                shape,
                children_julia,
                symbol.secondary_dimensions_npts,
                symbol._children_slices,
            )
        elif isinstance(symbol, PybammJuliaFunction):
            my_id = symbol.id
            child_ids = []
            for child in symbol.children:
                child_id = self._convert_tree_to_intermediate(child)
                child_ids.append(child_id)
            shape = symbol.shape
            name = symbol.name
            self._intermediate[my_id] = JuliaJuliaFunction(
                child_ids,
                my_id,
                shape,
                name
            )
        elif isinstance(symbol, pybamm.MatrixMultiplication):
            # Break down the binary tree
            id_left, id_right, my_id = self.break_down_binary(symbol)
            left_shape = self._intermediate[id_left].shape
            right_shape = self._intermediate[id_right].shape
            my_shape = (left_shape[0], right_shape[1])
            # Cache the result.
            self._intermediate[my_id] = JuliaMatrixMultiplication(
                id_left, id_right, my_id, my_shape
            )
        elif isinstance(symbol, pybamm.Multiplication) or isinstance(
            symbol, pybamm.Inner
        ):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaMultiplication(
                id_left, id_right, my_id, my_shape, "*"
            )
        elif isinstance(symbol, pybamm.Division):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaDivision(
                id_left, id_right, my_id, my_shape, "/"
            )
        elif isinstance(symbol, pybamm.Addition):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaAddition(
                id_left, id_right, my_id, my_shape, "+"
            )
        elif isinstance(symbol, pybamm.Subtraction):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaSubtraction(
                id_left, id_right, my_id, my_shape, "-"
            )
        elif isinstance(symbol, pybamm.Minimum):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaMinMax(
                id_left, id_right, my_id, my_shape, "min"
            )
        elif isinstance(symbol, pybamm.Maximum):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaMinMax(
                id_left, id_right, my_id, my_shape, "max"
            )
        elif isinstance(symbol, pybamm.Power):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaPower(
                id_left, id_right, my_id, my_shape, "^"
            )
        elif isinstance(symbol, pybamm.EqualHeaviside):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaBitwiseBinaryOperation(
                id_left, id_right, my_id, my_shape, "<="
            )
        elif isinstance(symbol, pybamm.NotEqualHeaviside):
            id_left, id_right, my_id = self.break_down_binary(symbol)
            my_shape = self.find_broadcastable_shape(id_left, id_right)
            self._intermediate[my_id] = JuliaBitwiseBinaryOperation(
                id_left, id_right, my_id, my_shape, "<"
            )
        elif isinstance(symbol, pybamm.Index):
            id_lower = self._convert_tree_to_intermediate(symbol.children[0])
            child_shape = self._intermediate[id_lower].shape
            child_ncols = child_shape[1]

            my_id = symbol.id
            index = symbol.index
            if isinstance(index, slice):
                if index.step is None:
                    shape = ((index.stop) - (index.start), child_ncols)
                elif isinstance(index.step, int):
                    shape = (
                        floor((index.stop - index.start) / index.step),
                        child_ncols,
                    )
            elif isinstance(index, int):
                shape = (1, child_ncols)

            self._intermediate[my_id] = JuliaIndex(id_lower, my_id, index, shape)
        elif isinstance(symbol, pybamm.Min) or isinstance(symbol, pybamm.Max):
            my_jl_name = symbol.julia_name
            my_shape = (1, 1)
            this_input = self._convert_tree_to_intermediate(symbol.children[0])
            my_id = symbol.id
            self._intermediate[my_id] = JuliaMinimumMaximum(
                my_jl_name, this_input, my_id, my_shape
            )
        elif isinstance(symbol, pybamm.Function):
            my_jl_name = symbol.julia_name
            my_shape = symbol.children[0].shape
            this_input = self._convert_tree_to_intermediate(symbol.children[0])
            my_id = symbol.id
            self._intermediate[my_id] = JuliaBroadcastableFunction(
                my_jl_name, this_input, my_id, my_shape
            )
        elif isinstance(symbol, pybamm.Negate):
            my_jl_name = "-"
            my_shape = symbol.children[0].shape
            this_input = self._convert_tree_to_intermediate(symbol.children[0])
            my_id = symbol.id
            self._intermediate[my_id] = JuliaNegation(
                my_jl_name, this_input, my_id, my_shape
            )
        elif isinstance(symbol, pybamm.Matrix):
            my_id = symbol.id
            value = symbol.evaluate()
            if value.shape == (1, 1):
                self._intermediate[my_id] = JuliaScalar(my_id, value)
            else:
                self._intermediate[my_id] = JuliaConstant(my_id, value)
        elif isinstance(symbol, pybamm.Vector):
            my_id = symbol.id
            value = symbol.evaluate()
            if value.shape == (1, 1):
                if isinstance(value, scipy.sparse._csr.csr_matrix):
                    value = value.toarray()
                self._intermediate[my_id] = JuliaScalar(my_id, value)
            else:
                self._intermediate[my_id] = JuliaConstant(my_id, value)
        elif isinstance(symbol, pybamm.Scalar):
            my_id = symbol.id
            value = symbol.evaluate()
            self._intermediate[my_id] = JuliaScalar(my_id, value)
        elif isinstance(symbol, pybamm.Time):
            my_id = symbol.id
            self._intermediate[my_id] = JuliaTime(my_id)
        elif isinstance(symbol, pybamm.PsuedoInputParameter):
            my_id = self._convert_tree_to_intermediate(symbol.children[0])
        elif isinstance(symbol, pybamm.InputParameter):
            my_id = symbol.id
            name = symbol.name
            self._intermediate[my_id] = JuliaInput(my_id, name)
        elif isinstance(symbol, pybamm.StateVector):
            my_id = symbol.id
            first_point = symbol.first_point
            last_point = symbol.last_point
            points = (first_point, last_point)
            shape = symbol.shape
            self._intermediate[my_id] = JuliaStateVector(my_id, points, shape)
        elif isinstance(symbol, pybamm.StateVectorDot):
            my_id = symbol.id
            first_point = symbol.first_point
            last_point = symbol.last_point
            points = (first_point, last_point)
            shape = symbol.shape
            self._intermediate[my_id] = JuliaStateVectorDot(my_id, points, shape)
        else:
            raise NotImplementedError(
                "Conversion to Julia not implemented for a symbol of type '{}'".format(
                    type(symbol)
                )
            )
        return my_id

    def find_broadcastable_shape(self, id_left, id_right):
        left_shape = self._intermediate[id_left].shape
        right_shape = self._intermediate[id_right].shape
        # check if either is a scalar
        if left_shape == (1, 1):
            return right_shape
        elif right_shape == (1, 1):
            return left_shape
        elif left_shape == right_shape:
            return left_shape
        elif (left_shape[0] == 1) & (right_shape[1] == 1):
            return (right_shape[0], left_shape[1])
        elif (right_shape[0] == 1) & (left_shape[1] == 1):
            return (left_shape[0], right_shape[1])
        elif (right_shape[0] == 1) & (right_shape[1] == left_shape[1]):
            return left_shape
        elif (left_shape[0] == 1) & (right_shape[1] == left_shape[1]):
            return right_shape
        elif (right_shape[1] == 1) & (right_shape[0] == left_shape[0]):
            return left_shape
        elif (left_shape[1] == 1) & (right_shape[0] == left_shape[0]):
            return right_shape
        else:
            raise NotImplementedError(
                "multiplication for the shapes youve requested doesnt work."
            )

    # Functions
    # Broadcastable functions have 1 input and 1 output, and the
    #  input and output have the same shape. The hard part is that
    # we have to know which is which and pybamm doesn't differentiate
    #  between the two. So, we have to do that with an if statement.

    # Cache and Const Creation
    def create_cache(self, symbol, cache_name=None):
        my_id = symbol.output
        cache_shape = self._intermediate[my_id].shape
        cache_id = self._cache_id
        self._cache_id += 1
        if cache_name is None:
            cache_name = "cache_{}".format(cache_id)
        
        if self._preallocate:
            if self._cache_type == "standard":
                if cache_shape[1] == 1:
                    cache_shape_st = "({})".format(cache_shape[0])
                else:
                    cache_shape_st = cache_shape
                self._cache_and_const_string += "{} = zeros{}\n".format(
                    cache_name, cache_shape_st
                )
                self._cache_dict[symbol.output] = cache_name
            elif self._cache_type == "dual":
                if cache_shape[1] == 1:
                    cache_shape_st = "({})".format(cache_shape[0])
                else:
                    cache_shape_st = cache_shape
                self._cache_and_const_string += (
                    "{}_init = dualcache(zeros{},12)\n".format(
                        cache_name, cache_shape_st
                    )
                )
                self._cache_initialization_string += (
                    "{} = PreallocationTools.get_tmp({}_init,(@view y[1:{}]))\n".format(
                        cache_name, cache_name, cache_shape[0]
                    )
                )
                self._cache_dict[symbol.output] = cache_name
            elif self._cache_type == "symbolic":
                if cache_shape[1] == 1:
                    cache_shape_st = "({})".format(cache_shape[0])
                else:
                    cache_shape_st = cache_shape
                self._cache_and_const_string += (
                    "   {}_init = symcache(zeros{},Vector{{Num}}(undef,{}))\n".format(
                        cache_name, cache_shape_st, cache_shape[0]
                    )
                )
                self._cache_initialization_string += (
                    "   {} = PyBaMM.get_tmp({}_init,(@view y[1:{}]))\n".format(
                        cache_name, cache_name, cache_shape[0]
                    )
                )
                self._cache_dict[symbol.output] = cache_name
            elif self._cache_type == "gpu":
                if cache_shape[1] == 1:
                    cache_shape_st = "({})".format(cache_shape[0])
                else:
                    cache_shape_st = cache_shape
                self._cache_and_const_string += "{} = CUDA.zeros{}\n".format(
                    cache_name, cache_shape_st
                )
                self._cache_dict[symbol.output] = cache_name
            else:
                raise NotImplementedError(
                    "The cache type you've specified has not yet been implemented"
                )
            return self._cache_dict[my_id]
        else:
            self._cache_dict[symbol.output] = cache_name
            return self._cache_dict[symbol.output]

    def create_const(self, symbol, cache_name=None):
        my_id = symbol.output
        const_id = self._const_id + 1
        self._const_id = const_id
        if cache_name is None:
            const_name = "const_{}".format(const_id)
        else:
            const_name = cache_name
        self._const_dict[my_id] = const_name
        mat_value = symbol.value
        val_line = self.write_const(mat_value)
        if self._cache_type == "gpu":
            const_line = const_name + " = cu({})\n".format(val_line)
        else:
            const_line = const_name + " = {}\n".format(val_line)
        self._cache_and_const_string += const_line
        return 0

    def write_const(self, value):
        if isinstance(value, np.ndarray):
            val_string = value
        elif isinstance(value, scipy.sparse._csr.csr_matrix):
            row, col, data = scipy.sparse.find(value)
            m, n = value.shape
            np.set_printoptions(
                threshold=max(np.get_printoptions()["threshold"], len(row) + 10)
            )

            val_string = "sparse({}, {}, {}{}, {}, {})".format(
                np.array2string(row + 1, separator=","),
                np.array2string(col + 1, separator=","),
                self._type,
                np.array2string(data, separator=","),
                m,
                n,
            )
        else:
            raise NotImplementedError("attempted to write an unsupported const")
        return val_string

    def clear(self):
        self._intermediate = OrderedDict()
        self._function_string = ""
        self._cache_dict = OrderedDict()
        self._cache_and_const_string = ""
        self._const_dict = OrderedDict()
        self._dag = {}
        self._code = {}
        self._cache_id = 0
        self._const_id = 0

    def write_function(self):
        ts = graphlib.TopologicalSorter(self._dag)
        if self._parallel is None:
            for node in ts.static_order():
                code = self._code.get(node)
                if code is not None:
                    self._function_string += self._code[node]
        elif self._parallel == "Threads":
            ts.prepare()
            while ts.is_active():
                self._function_string += "@sync begin\n"
                for node in ts.get_ready():
                    code = self._code.get(node)
                    if code is not None:
                        code = code[0:-1]
                        code = "Threads.@spawn begin " + code + " end\n"
                        self._function_string += code
                    ts.done(node)
                self._function_string += "end\n"
        elif self._parallel == "legacy-serial":
            pass
        else:
            raise NotImplementedError()

    def write_black_box(self, funcname):
        top = self._intermediate[next(reversed(self._intermediate))]
        if len(self.outputs) != 1:
            raise NotImplementedError(
                "only 1 output is allowed!"
            )
        if not self._preallocate:
            raise NotImplementedError(
                "black box only supports preallocation."
            )
        #this will automatically be in place with the correct function name
        top_var_name = top._convert_intermediate_to_code(self, inline=False, cache_name=self.outputs[0])
        
        #still need to write the function.
        self.write_function()
        self._cache_and_const_string = (
            "begin\n{} = let \n".format(funcname) + self._cache_and_const_string
        )
        
        #No need to write a cache since it's in the input.
        self._cache_and_const_string = remove_lines_with(
            self._cache_and_const_string, top_var_name
        )

        #may need to modify this logic a bit in the future.
        if "p" in self.inputs:
            parameter_string = ""
            for parameter in self.input_parameter_order:
                parameter_string += "{},".format(parameter)
            parameter_string += "= p\n"
            self._function_string = parameter_string + self._function_string
        
        #same as _easy
        self._function_string = (
            self._cache_initialization_string + self._function_string
        )

        #no support for not preallocating. (for now)
        self._function_string += "\n   return nothing\nend\nend\nend"
        header_string = "@inbounds function " + funcname + "_with_consts" + "(" + top_var_name
        for this_input in self.inputs:
            header_string = header_string + "," + this_input
        header_string +=")\n"
        self._function_string = header_string + self._function_string
        return 0

    # Just get something working here, so can start actual testing
    def write_function_easy(self, funcname, inline=True):
        # start with the closure
        top = self._intermediate[next(reversed(self._intermediate))]
        # this line actually writes the code
        top_var_name = top._convert_intermediate_to_code(self, inline=False)
        # if parallel is true, we haven't actually written the function yet
        self.write_function()
        # write the cache initialization
        self._cache_and_const_string = (
            "begin\n{} = let \n".format(funcname) + self._cache_and_const_string
        )
        if len(self._intermediate) > 1:
            self._cache_and_const_string = remove_lines_with(
                self._cache_and_const_string, top_var_name
            )
            self._cache_initialization_string = remove_lines_with(
                self._cache_initialization_string, top_var_name
            )
        my_shape = top.shape
        if len(self.input_parameter_order) != 0:
            parameter_string = ""
            for parameter in self.input_parameter_order:
                parameter_string += "{},".format(parameter)
            parameter_string += "= p\n"
            self._function_string = parameter_string + self._function_string
        self._function_string = (
            self._cache_initialization_string + self._function_string
        )
        if self._preallocate:
            self._function_string += "\n   return nothing\n"
        else:
            self._function_string += "\n   return {}\n".format(top_var_name)
        if my_shape[1] != 1:
            self._function_string = self._function_string.replace(top_var_name, "J")
            self._function_string += "end\nend\nend"
            self._function_string = (
                "@inbounds function {}(J, y, p, t)\n".format(funcname + "_with_consts")
                + self._function_string
            )
        elif self._dae_type == "semi-explicit":
            self._function_string = self._function_string.replace(top_var_name, "dy")
            self._function_string += "end\nend\nend"
            self._function_string = (
                "@inbounds function {}(dy, y, p, t)\n".format(funcname + "_with_consts")
                + self._function_string
            )
        elif self._dae_type == "implicit":
            self._function_string = self._function_string.replace(top_var_name, "out")
            self._function_string += "end\nend\nend"
            self._function_string = (
                "@inbounds function {}(out, dy, y, p, t)\n".format(
                    funcname + "_with_consts"
                )
                + self._function_string
            )
        return 0

    # this function will be the top level.
    def convert_tree_to_intermediate(self, symbol, len_rhs=None):
        if isinstance(symbol, pybamm.PybammJuliaFunction):
            #need to hash this out a bit more.
            self._black_box = True
            self.outputs = ["out"]
            self.inputs = []
            self.funcname = symbol.name
            #process inputs: input types can be StateVectors,
            #StateVectorDots, parameters, time, and psuedo
            # parameters. 
        elif self._dae_type == "implicit":
            symbol_minus_dy = []
            end = 0
            for child in symbol.orphans:
                start = end
                end += child.size
                if end <= len_rhs:
                    symbol_minus_dy.append(
                        child - pybamm.StateVectorDot(slice(start, end))
                    )
                else:
                    symbol_minus_dy.append(child)
            symbol = pybamm.numpy_concatenation(*symbol_minus_dy)
        if isinstance(symbol, pybamm.PybammJuliaFunction):
            self._convert_tree_to_intermediate(symbol.expr)
        else:
            self._convert_tree_to_intermediate(symbol)
        return 0

    # rework this at some point
    def build_julia_code(self, funcname="f", inline=True):
        # get top node of tree
        if self._black_box:
            funcname = self.funcname
            self.write_black_box(funcname)
        else:
            self.write_function_easy(funcname, inline=inline)
        string = self._cache_and_const_string + self._function_string
        return string

#this is a bit of a weird one, may change it at some point
class JuliaJuliaFunction(object):
    def __init__(self, inputs, output, shape, name):
        self.inputs = inputs
        self.output = output
        self.shape = shape
        self.name = name
    
    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=False, cache_name=None):
        if converter.cache_exists(self.output, self.inputs):
            return converter._cache_dict[self.output]
        result_var_name = converter.create_cache(self, cache_name = cache_name)

        input_var_names = []
        for input in self.inputs:
            input_var_names.append(
                converter._intermediate[input]._convert_intermediate_to_code(converter, inline=inline)
            )
        result_var_name = converter._cache_dict[self.output]
        code = "{}({}".format(self.name, result_var_name)
        for input in input_var_names:
            code = code+","+input
        code = code+")\n"

        #black box always generates a cache.
        self.generate_code_and_dag(converter, code)
        return result_var_name
    
    def generate_code_and_dag(self, converter, code):
        converter._code[self.output] = code
        converter._dag[self.output] = set(self.inputs)
        if converter._parallel == "legacy-serial":
            converter._function_string += code


# BINARY OPERATORS: NEED TO DEFINE ONE FOR EACH MULTIPLE DISPATCH
class JuliaBinaryOperation(object):
    def __init__(self, left_input, right_input, output, shape):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape

    def get_binary_inputs(self, converter: JuliaConverter, inline=True):
        left_input_var_name = converter._intermediate[
            self.left_input
        ]._convert_intermediate_to_code(converter, inline=inline)
        right_input_var_name = converter._intermediate[
            self.right_input
        ]._convert_intermediate_to_code(converter, inline=inline)
        return left_input_var_name, right_input_var_name

    def generate_code_and_dag(self, converter, code):
        converter._code[self.output] = code
        l_id = converter._intermediate[self.left_input].output
        r_id = converter._intermediate[self.right_input].output
        converter._dag[self.output] = {l_id, r_id}
        if converter._parallel == "legacy-serial":
            converter._function_string += code


# MatMul and Inner Product are not really the same as the bitwisebinary operations.
class JuliaMatrixMultiplication(JuliaBinaryOperation):
    def __init__(self, left_input, right_input, output, shape):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=False, cache_name=None):
        if converter.cache_exists(self.output, [self.left_input, self.right_input]):
            return converter._cache_dict[self.output]
        result_var_name = converter.create_cache(self, cache_name = cache_name)
        left_input_var_name, right_input_var_name = self.get_binary_inputs(
            converter, inline=False
        )
        result_var_name = converter._cache_dict[self.output]
        if converter._preallocate:
            code = "mul!({},{},{})\n".format(
                result_var_name, left_input_var_name, right_input_var_name
            )
        else:
            code = "{} = {} * {}\n".format(
                result_var_name, left_input_var_name, right_input_var_name
            )
        # mat-mul is always creating a cache
        self.generate_code_and_dag(converter, code)
        return result_var_name


# Includes Addition, subtraction, multiplication, division, power, minimum, and maximum
class JuliaBitwiseBinaryOperation(JuliaBinaryOperation):
    def __init__(self, left_input, right_input, output, shape, operator):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape
        self.operator = operator

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, [self.left_input, self.right_input]):
            return converter._cache_dict[self.output]
        inline = inline & converter._inline
        if not inline:
            result_var_name = converter.create_cache(self, cache_name=cache_name)
            left_input_var_name, right_input_var_name = self.get_binary_inputs(
                converter, inline=True
            )
            if converter._preallocate:
                code = "@. {} = {} {} {}\n".format(
                    result_var_name,
                    left_input_var_name,
                    self.operator,
                    right_input_var_name,
                )
            else:
                code = "{} = @. ( {} {} {})\n".format(
                    result_var_name,
                    left_input_var_name,
                    self.operator,
                    right_input_var_name,
                )
            self.generate_code_and_dag(converter, code)
        elif inline:
            left_input_var_name, right_input_var_name = self.get_binary_inputs(
                converter, inline=True
            )
            result_var_name = "({} {} {})".format(
                left_input_var_name, self.operator, right_input_var_name
            )
        return result_var_name


# PREALLOCATING STOPPED HERE
class JuliaAddition(JuliaBitwiseBinaryOperation):
    pass


class JuliaSubtraction(JuliaBitwiseBinaryOperation):
    pass


class JuliaMultiplication(JuliaBitwiseBinaryOperation):
    pass


class JuliaDivision(JuliaBitwiseBinaryOperation):
    pass


class JuliaPower(JuliaBitwiseBinaryOperation):
    pass


# MinMax is special because it does both min and max.
# Could be folded into JuliaBitwiseBinaryOperation once I do that
class JuliaMinMax(JuliaBinaryOperation):
    def __init__(self, left_input, right_input, output, shape, name):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape
        self.name = name

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, [self.left_input, self.right_input]):
            return converter._cache_dict[self.output]
        inline = inline & converter._inline

        if not inline:
            result_var_name = converter.create_cache(self)
            left_input_var_name, right_input_var_name = self.get_binary_inputs(
                converter, inline=True
            )
            if converter._preallocate:
                code = "@. {} = {}({},{})\n".format(
                    result_var_name,
                    self.name,
                    left_input_var_name,
                    right_input_var_name,
                )
            else:
                code = "{} = {}.({},{})\n".format(
                    result_var_name,
                    self.name,
                    left_input_var_name,
                    right_input_var_name,
                )
            self.generate_code_and_dag(converter, code)
        elif inline:
            left_input_var_name, right_input_var_name = self.get_binary_inputs(
                converter, inline=True
            )
            result_var_name = "{}({},{})".format(
                self.name, left_input_var_name, right_input_var_name
            )
        return result_var_name


# FUNCTIONS
# All Functions Return the same number of arguments
# they take in, except for minimum and maximum.
class JuliaFunction(object):
    pass


class JuliaBroadcastableFunction(JuliaFunction):
    def __init__(self, name, this_input, output, shape):
        self.name = name
        self.input = this_input
        self.output = output
        self.shape = shape

    def generate_code_and_dag(self, converter: JuliaConverter, code):
        converter._code[self.output] = code
        converter._dag[self.output] = {converter._intermediate[self.input].output}
        if converter._parallel == "legacy-serial":
            converter._function_string += code

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, [self.input]):
            return converter._cache_dict[self.output]
        inline = inline & converter._inline
        if not inline:
            result_var_name = converter.create_cache(self, cache_name=cache_name)
            input_var_name = converter._intermediate[
                self.input
            ]._convert_intermediate_to_code(converter, inline=True)
            if converter._preallocate:
                code = "@. {} = {}({})\n".format(
                    result_var_name, self.name, input_var_name
                )
            else:
                code = "{} = {}.({})\n".format(
                    result_var_name, self.name, input_var_name
                )
            self.generate_code_and_dag(converter, code)
        else:
            # assume an @. has already been issued
            input_var_name = converter._intermediate[
                self.input
            ]._convert_intermediate_to_code(converter, inline=True)
            result_var_name = "({}({}))".format(self.name, input_var_name)
        return result_var_name


class JuliaNegation(JuliaBroadcastableFunction):
    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, [self.input]):
            return converter._cache_dict[self.output]
        inline = inline & converter._inline

        if not inline:
            result_var_name = converter.create_cache(self, cache_name=cache_name)
            input_var_name = converter._intermediate[
                self.input
            ]._convert_intermediate_to_code(converter, inline=True)
            if converter._preallocate:
                code = "@. {} = - {}\n".format(result_var_name, input_var_name)
            else:
                code = "{} = -{}\n".format(result_var_name, input_var_name)
            self.generate_code_and_dag(converter, code)
        else:
            input_var_name = converter._intermediate[
                self.input
            ]._convert_intermediate_to_code(converter, inline=True)
            result_var_name = "(- {})".format(input_var_name)
        return result_var_name


class JuliaMinimumMaximum(JuliaBroadcastableFunction):
    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, [self.input]):
            return converter._cache_dict[self.output]
        result_var_name = converter.create_cache(self, cache_name=cache_name)
        input_var_name = converter._intermediate[
            self.input
        ]._convert_intermediate_to_code(converter, inline=False)
        if converter._preallocate:
            code = "{} .= {}({})\n".format(result_var_name, self.name, input_var_name)
        else:
            code = "{} = {}({})\n".format(result_var_name, self.name, input_var_name)
        self.generate_code_and_dag(converter, code)
        return result_var_name


# Index is a little weird, so it just sits on its own.
class JuliaIndex(object):
    def __init__(self, this_input, output, index, shape):
        self.input = this_input
        self.output = output
        self.index = index
        self.shape = shape

    def generate_code_and_dag(self, converter: JuliaConverter, code):
        input_id = converter._intermediate[self.input].output
        converter._code[self.output] = code
        converter._dag[self.output] = {input_id}
        if converter._parallel == "legacy-serial":
            converter._function_string += code

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, [self.input]):
            return converter._cache_dict[self.output]
        index = self.index
        inline = inline & converter._inline
        if self.shape[1] == 1:
            right_parenthesis = "]"
        else:
            right_parenthesis = ",:]"
        if inline:
            input_var_name = converter._intermediate[
                self.input
            ]._convert_intermediate_to_code(converter, inline=False)
            if isinstance(index, int):
                return "{}[{}]".format(input_var_name, index + 1)
            elif isinstance(index, slice):
                if index.step is None:
                    return "(@view {}[{}:{}{})".format(
                        input_var_name, index.start + 1, index.stop, right_parenthesis
                    )
                elif isinstance(index.step, int):
                    return "(@view {}[{}:{}:{}{})".format(
                        input_var_name,
                        index.start + 1,
                        index.step,
                        index.stop,
                        right_parenthesis,
                    )
                else:
                    raise NotImplementedError("Step has to be an integer.")
            else:
                raise NotImplementedError("Step must be a slice or an int")
        else:
            result_var_name = converter.create_cache(self)
            input_var_name = converter._intermediate[
                self.input
            ]._convert_intermediate_to_code(converter, inline=False)
            if isinstance(index, int):
                code = "@. {} = {}[{}{}".format(
                    result_var_name, input_var_name, index + 1, right_parenthesis
                )
            elif isinstance(index, slice):
                if index.step is None:
                    code = "@. {} = (@view {}[{}:{}{})\n".format(
                        result_var_name,
                        input_var_name,
                        index.start + 1,
                        index.stop,
                        right_parenthesis,
                    )
                elif isinstance(index.step, int):
                    code = "@. {} = (@view {}[{}:{}:{}{})\n".format(
                        result_var_name,
                        input_var_name,
                        index.start + 1,
                        index.step,
                        index.stop,
                        right_parenthesis,
                    )
                else:
                    raise NotImplementedError("Step has to be an integer.")
            else:
                raise NotImplementedError("Step must be a slice or an int")
            self.generate_code_and_dag(converter, code)
            return result_var_name


# Values and Constants -- I will need to change this to inputs, due to t, y, and p.
class JuliaValue(object):
    def generate_code_and_dag(self, converter: JuliaConverter):
        pass


class JuliaConstant(JuliaValue):
    def __init__(self, my_id, value):
        self.output = my_id
        self.value = value
        self.shape = value.shape

    def _convert_intermediate_to_code(self, converter, inline=True, cache_name=None):
        converter.create_const(self, cache_name=cache_name)
        self.generate_code_and_dag(converter)
        return converter._const_dict[self.output]


class JuliaStateVector(JuliaValue):
    def __init__(self, my_id, loc, shape):
        self.output = my_id
        self.loc = loc
        self.shape = shape

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        start = self.loc[0] + 1
        end = self.loc[1]
        self.generate_code_and_dag(converter)
        if start == end:
            return "(y[{}])".format(start)
        else:
            return "(@view y[{}:{}])".format(start, end)


class JuliaStateVectorDot(JuliaStateVector):
    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        start = self.loc[0] + 1
        end = self.loc[1]
        self.generate_code_and_dag(converter)
        if start == end:
            return "(dy[{}])".format(start)
        else:
            return "(@view dy[{}:{}])".format(start, end)


class JuliaScalar(JuliaConstant):
    def __init__(self, my_id, value):
        self.output = my_id
        self.value = float(value)
        self.shape = (1, 1)

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        self.generate_code_and_dag(converter)
        return self.value


class JuliaTime(JuliaScalar):
    def __init__(self, my_id):
        self.output = my_id
        self.shape = (1, 1)

    def _convert_intermediate_to_code(self, converter, inline=True, cache_name=None):
        self.generate_code_and_dag(converter)
        return "t"


class JuliaInput(JuliaScalar):
    def __init__(self, my_id, name):
        self.output = my_id
        self.shape = (1, 1)
        self.name = name

    def _convert_intermediate_to_code(self, converter, inline=True, cache_name=None):
        self.generate_code_and_dag(converter)
        return self.name


# CONCATENATIONS
class JuliaConcatenation(object):
    def __init__(self, output, shape, children):
        self.output = output
        self.shape = shape
        self.children = children

    def generate_code_and_dag(self, converter: JuliaConverter, code):
        ids = set(converter._intermediate[child].output for child in self.children)
        converter._dag[self.output] = ids
        converter._code[self.output] = code
        if converter._parallel == "legacy-serial":
            converter._function_string += code

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, self.children):
            return converter._cache_dict[self.output]
        num_cols = self.shape[1]
        my_name = converter.create_cache(self, cache_name=cache_name)

        # assume we don't have tensors. Already asserted
        # concatenations have to have the same width.
        if num_cols == 1:
            right_parenthesis = "]"
            vec = True
        else:
            right_parenthesis = ",:]"
            vec = False

        # do the 0th one outside of the loop to initialize
        child = self.children[0]
        child_var = converter._intermediate[child]
        child_var_name = child_var._convert_intermediate_to_code(converter, inline=True)
        start_row = 1
        if child_var.shape[0] == 0:
            end_row = 1
            code = ""
        elif child_var.shape[0] == 1:
            end_row = 1
            if converter._preallocate:
                if vec:
                    code = "@. {}[{}:{}{} =  {}\n".format(
                        my_name, start_row, start_row, right_parenthesis, child_var_name
                    )
                else:
                    code = " {}[{}{} = {}\n".format(
                        my_name, start_row, right_parenthesis, child_var_name
                    )
            else:
                code = "{} = vcat({}".format(my_name, child_var_name)
        else:
            start_row = 1
            end_row = child_var.shape[0]
            if converter._preallocate:
                code = "@. {}[{}:{}{} = {}\n".format(
                    my_name, start_row, end_row, right_parenthesis, child_var_name
                )
            else:
                code = " {} = vcat( {} ".format(my_name, child_var_name)
        counter = "b"
        for child in self.children[1:]:
            child_var = converter._intermediate[child]
            child_var_name = child_var._convert_intermediate_to_code(
                converter, inline=True
            )
            counter = chr(ord(counter) + 1)
            if child_var.shape[0] == 0:
                continue
            elif child_var.shape[0] == 1:
                start_row = end_row + 1
                end_row = start_row
                if converter._preallocate:
                    if vec:
                        code += "@. {}[{}:{}{} =  {}\n".format(
                            my_name, start_row, start_row, right_parenthesis, child_var_name
                        )
                    else:
                        code += "@. {}[{}{} = {} \n".format(
                            my_name, start_row, right_parenthesis, child_var_name
                        )
                elif child == self.children[-1]:
                    code += ",{} )\n".format(child_var_name)
                else:
                    code += ", {} ".format(child_var_name)
            else:
                start_row = end_row + 1
                end_row = start_row + child_var.shape[0] - 1
                if converter._preallocate:
                    code += "@. {}[{}:{}{} = {} \n".format(
                        my_name, start_row, end_row, right_parenthesis, child_var_name
                    )
                elif child == self.children[-1]:
                    code += ",{} )\n".format(child_var_name)
                else:
                    code += ", {} ".format(child_var_name)
        self.generate_code_and_dag(converter, code)
        return my_name


class JuliaNumpyConcatenation(JuliaConcatenation):
    pass


# NOTE: CURRENTLY THIS BEHAVES EXACTLY LIKE NUMPYCONCATENATION
class JuliaSparseStack(JuliaConcatenation):
    pass


class JuliaDomainConcatenation(JuliaConcatenation):
    def __init__(
        self, output, shape, children, secondary_dimension_npts, children_slices
    ):
        self.output = output
        self.shape = shape
        self.children = children
        self.secondary_dimension_npts = secondary_dimension_npts
        self.children_slices = children_slices

    def _convert_intermediate_to_code(self, converter: JuliaConverter, inline=True, cache_name=None):
        if converter.cache_exists(self.output, self.children):
            return converter._cache_dict[self.output]
        num_cols = self.shape[1]
        result_var_name = converter.create_cache(self, cache_name=cache_name)

        # assume we don't have tensors. Already asserted
        # that concatenations have to have the same width.
        if num_cols == 1:
            right_parenthesis = "]"
        else:
            right_parenthesis = ",:]"
        # do the 0th one outside of the loop to initialize
        end_row = 0
        code = ""
        if self.secondary_dimension_npts == 1:
            for c in range(len(self.children)):
                child = converter._intermediate[self.children[c]]
                child_var_name = child._convert_intermediate_to_code(
                    converter, inline=True
                )
                this_slice = list(self.children_slices[c].values())[0][0]
                start = this_slice.start
                stop = this_slice.stop
                start_row = end_row + 1
                end_row = start_row + (stop - start) - 1
                if converter._preallocate:
                    code += "@. {}[{}:{}{} = {} \n".format(
                        result_var_name,
                        start_row,
                        end_row,
                        right_parenthesis,
                        child_var_name,
                    )
                else:
                    if c == 0:
                        code += "{} = vcat( {} ".format(result_var_name, child_var_name)
                    elif c == len(self.children) - 1:
                        code += ", {} )\n".format(child_var_name)
                    else:
                        code += ", {}".format(child_var_name)

        else:
            num_chil = len(self.children)
            for i in range(self.secondary_dimension_npts):
                for c in range(num_chil):
                    child = converter._intermediate[self.children[c]]
                    child_var_name = child._convert_intermediate_to_code(
                        converter, inline=True
                    )
                    this_slice = list(self.children_slices[c].values())[0][i]
                    start = this_slice.start
                    stop = this_slice.stop
                    start_row = end_row + 1
                    end_row = start_row + (stop - start) - 1
                    if converter._preallocate:
                        code += "@. {}[{}:{}{} = (@view {}[{}:{}{})\n".format(
                            result_var_name,
                            start_row,
                            end_row,
                            right_parenthesis,
                            child_var_name,
                            start + 1,
                            stop,
                            right_parenthesis,
                        )
                    else:
                        if (c == 0) & (i == 0):
                            code += "{} = vcat((@view {}[{}:{}{})".format(
                                result_var_name,
                                child_var_name,
                                start + 1,
                                stop,
                                right_parenthesis,
                            )
                        elif (c == len(self.children) - 1) & (
                            i == self.secondary_dimension_npts - 1
                        ):
                            code += ",(@view {}[{}:{}{}))\n".format(
                                child_var_name, start + 1, stop, right_parenthesis
                            )
                        else:
                            code += ",(@view {}[{}:{}{})".format(
                                child_var_name, start + 1, stop, right_parenthesis
                            )
        self.generate_code_and_dag(converter, code)
        return result_var_name
