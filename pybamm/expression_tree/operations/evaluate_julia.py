#
# Compile a PyBaMM expression tree to Julia Code
#
import pybamm
import numpy as np
import numpy
from scipy import special
import scipy
from collections import OrderedDict
from multimethod import multimethod
from math import floor

def is_constant_and_can_evaluate(symbol):
    """
    Returns True if symbol is constant and evaluation does not raise any errors.
    Returns False otherwise.
    An example of a constant symbol that cannot be "evaluated" is PrimaryBroadcast(0).
    """
    if symbol.is_constant():
        try:
            symbol.evaluate()
            return True
        except NotImplementedError:
            return False
    else:
        return False

#BINARY OPERATORS: NEED TO DEFINE ONE FOR EACH MULTIPLE DISPATCH
class JuliaBinaryOperation(object):
    def __init__(self,left_input,right_input,output,shape):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape

#MatMul and Inner Product are not really the same as the bitwisebinary operations.
class JuliaMatrixMultiplication(JuliaBinaryOperation):
    def __init__(self,left_input,right_input,output,shape):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape

class JuliaBitwiseBinaryOperation(JuliaBinaryOperation):
    def __init__(self,left_input,right_input,output,shape,operator):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape
        self.operator = operator

class JuliaAddition(JuliaBinaryOperation):
    pass

class JuliaSubtraction(JuliaBinaryOperation):
    pass

class JuliaMultiplication(JuliaBinaryOperation):
    pass

class JuliaDivision(JuliaBinaryOperation):
    pass

class JuliaPower(JuliaBinaryOperation):
    pass

#MinMax is special because it does both min and max. Could be folded into JuliaBitwiseBinaryOperation once I do that
class JuliaMinMax(JuliaBinaryOperation):
    def __init__(self,left_input,right_input,output,shape,name):
        self.left_input = left_input
        self.right_input = right_input
        self.output = output
        self.shape = shape
        self.name = name

#FUNCTIONS
##All Functions Return the same number of arguments they take in, except for minimum and maximum.
class JuliaFunction(object):
    pass

class JuliaBroadcastableFunction(JuliaFunction):
    def __init__(self,name,input,output,shape):
        self.name = name
        self.input = input
        self.output = output
        self.shape = shape

class JuliaNegation(JuliaBroadcastableFunction):
    pass

class JuliaMinimumMaximum(JuliaBroadcastableFunction):
    pass


#Index is a little weird, so it just sits on its own.
class JuliaIndex(object):
    def __init__(self,input,output,index):
        self.input = input
        self.output = output
        self.index = index
        if type(index) is slice:
            if index.step == None:
                self.shape = ((index.stop)-(index.start),1)
            elif type(index.step) == int:
                self.shape = (floor((index.stop-index.start)/index.step),1)
            else:
                print(index.step)
                raise NotImplementedError("asldhfjwaes")
        elif type(index) is int:
            self.shape = (1,1)
        else:
            raise NotImplementedError("index must be slice or int")



#Values and Constants -- I will need to change this to inputs, due to t, y, and p.
class JuliaValue(object):
    pass

class JuliaConstant(JuliaValue):
    def __init__(self,id,value):
        self.id = id
        self.value = value
        self.shape = value.shape

class JuliaStateVector(JuliaValue):
    def __init__(self,id,loc,shape):
        self.id = id
        self.loc = loc
        self.shape = shape

class JuliaStateVectorDot(JuliaStateVector):
    pass

class JuliaScalar(JuliaConstant):
    def __init__(self,id,value):
        self.id = id
        self.value = float(value)
        self.shape = (1,1)

class JuliaTime(JuliaScalar):
    def __init__(self,id):
        self.id = id
        self.shape = (1,1)

class JuliaInput(JuliaScalar):
    def __init__(self,id,name):
        self.id = id
        self.shape = (1,1)
        self.name = name



#CONCATENATIONS
class JuliaConcatenation(object):
    def __init__(self,output,shape,children):
        self.output = output
        self.shape = shape
        self.children = children

class JuliaNumpyConcatenation(JuliaConcatenation):
    pass

#NOTE: CURRENTLY THIS BEHAVES EXACTLY LIKE NUMPYCONCATENATION
class JuliaSparseStack(JuliaConcatenation):
    pass


class JuliaDomainConcatenation(JuliaConcatenation):
    def __init__(self,output,shape,children,secondary_dimension_npts,children_slices):
        self.output = output
        self.shape = shape
        self.children = children
        self.secondary_dimension_npts = secondary_dimension_npts
        self.children_slices = children_slices




class JuliaConverter(object):
    def __init__(self,ismtk=False,cache_type="standard",jacobian_type="analytical",preallocate=True,dae_type="semi-explicit"): 
        assert not ismtk

        #Characteristics
        self._cache_type = cache_type
        self._ismtk=ismtk
        self._jacobian_type=jacobian_type
        self._preallocate=preallocate
        self._dae_type = dae_type

        self._type = "Float64"
        #"Caches"
        #Stores Constants to be Declared in the initial cache
        #insight: everything is just a line of code
        
        #INTERMEDIATE: A List of Stuff to do. Keys are ID's and lists are variable names.
        self._intermediate = OrderedDict()

        #Cache Dict and Const Dict Host Julia Variable Names to be used to generate the code. 
        self._cache_dict = OrderedDict()
        self._const_dict = OrderedDict()

        self._parameter_dict = OrderedDict()
        
        self._cache_id = 0
        self._const_id = 0
        
        self._cache_and_const_string = ""
        self.function_definition = ""
        self._function_string = ""
        self._return_string = ""
    
    #know where to go to find a variable. this could be smoother, there will need to be a ton of boilerplate here.
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaConcatenation):
        return self._cache_dict[julia_symbol.output]
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaMinimumMaximum):
        return self._cache_dict[julia_symbol.output]
    
    @multimethod
    def get_result_variable_name(self, julia_symbol:JuliaBinaryOperation):
        return self._cache_dict[julia_symbol.output]
    
    @multimethod 
    def get_result_variable_name(self,julia_symbol:JuliaConstant):
        return self._const_dict[julia_symbol.id]
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaScalar):
        return julia_symbol.value
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaTime):
        return "t"
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaInput):
        return julia_symbol.name
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaStateVector):
        start = julia_symbol.loc[0]+1
        end = julia_symbol.loc[1]
        if start==end:
            return "y[{}]".format(start)
        else:
            return "y[{}:{}]".format(start,end)
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaStateVectorDot):
        start = julia_symbol.loc[0]+1
        end = julia_symbol.loc[1]
        if start==end:
            return "dy[{}]".format(start)
        else:
            return "dy[{}:{}]".format(start,end)
    
    @multimethod
    def get_result_variable_name(self,julia_symbol:JuliaBroadcastableFunction):
        return self._cache_dict[julia_symbol.output]

    @multimethod 
    def get_result_variable_name(self,julia_symbol:JuliaIndex):
        lower_var = self.get_result_variable_name(self._intermediate[julia_symbol.input])
        index = julia_symbol.index
        if type(index) is int:
            return "{}[{}]".format(lower_var,index+1)
        elif type(index) is slice:
            if index.step is None:
                return "{}[{}:{}]".format(lower_var,index.start+1,index.stop)
            elif type(index.step) is int:
                return "{}[{}:{}:{}]".format(lower_var,index.start+1,index.step,index.stop)
            else:
                raise NotImplementedError("Step has to be an integer.")
        else:
            raise NotImplementedError("Step must be a slice or an int")
    
    #This function breaks down and analyzes any binary tree. Will fail if used on a non-binary tree.
    def break_down_binary(self,symbol):
        #Check for constant
        #assert not is_constant_and_can_evaluate(symbol)
        
        #We know that this should only have 2 children
        assert len(symbol.children)==2

        #take care of the kids first (this is recursive but multiple-dispatch recursive which is cool)
        id_left = self._convert_tree_to_intermediate(symbol.children[0])
        id_right = self._convert_tree_to_intermediate(symbol.children[1])
        my_id = symbol.id
        return id_left,id_right,my_id
    
    def break_down_concatenation(self,symbol):
        child_ids = []
        for child in symbol.children:
            child_id = self._convert_tree_to_intermediate(child)
            child_ids.append(child_id)
        first_id = child_ids[0]
        num_cols = self._intermediate[first_id].shape[1]
        num_rows = 0
        for child_id in child_ids:
            child_shape = self._intermediate[child_id].shape
            assert num_cols == child_shape[1]
            num_rows+=child_shape[0]
        shape = (num_rows,num_cols)
        return child_ids,shape

    
    #Convert-Trees go here  

    #Binary trees constructors. All follow the pattern of mat-mul. They need to find their shapes, assuming that the shapes of the nodes one level below them in the expression tree have already been computed.
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.NumpyConcatenation):
        my_id = symbol.id
        children_julia,shape = self.break_down_concatenation(symbol)
        self._intermediate[my_id] = JuliaNumpyConcatenation(my_id,shape,children_julia)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.SparseStack):
        my_id = symbol.id
        children_julia,shape = self.break_down_concatenation(symbol)
        self._intermediate[my_id] = JuliaSparseStack(my_id,shape,children_julia)
        return my_id

    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.DomainConcatenation):
        my_id = symbol.id
        children_julia,shape = self.break_down_concatenation(symbol)
        self._intermediate[my_id] = JuliaDomainConcatenation(my_id,shape,children_julia,symbol.secondary_dimensions_npts,symbol._children_slices)
        return my_id


    @multimethod
    def _convert_tree_to_intermediate(self,symbol: pybamm.MatrixMultiplication):
        #Break down the binary tree
        id_left,id_right,my_id = self.break_down_binary(symbol)
        left_shape = self._intermediate[id_left].shape
        right_shape = self._intermediate[id_right].shape
        my_shape = (left_shape[0],right_shape[1])
        #Cache the result.
        self._intermediate[my_id] = JuliaMatrixMultiplication(id_left,id_right,my_id,my_shape)
        return my_id
    
    @multimethod 
    def _convert_tree_to_intermediate(self,symbol:pybamm.Multiplication):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaMultiplication(id_left,id_right,my_id,my_shape)
        return my_id
    
    #Apparently an inner product is a hadamard product in pybamm
    @multimethod 
    def _convert_tree_to_intermediate(self,symbol:pybamm.Inner):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaMultiplication(id_left,id_right,my_id,my_shape)
        return my_id
    
    @multimethod 
    def _convert_tree_to_intermediate(self,symbol:pybamm.Division):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaDivision(id_left,id_right,my_id,my_shape)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol: pybamm.Addition):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaAddition(id_left,id_right,my_id,my_shape)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol: pybamm.Subtraction):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaSubtraction(id_left,id_right,my_id,my_shape)
        return my_id

    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Minimum):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaMinMax(id_left,id_right,my_id,my_shape,"min.")
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Maximum):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaMinMax(id_left,id_right,my_id,my_shape,"max.")
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Power):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaPower(id_left,id_right,my_id,my_shape)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.EqualHeaviside):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaBitwiseBinaryOperation(id_left,id_right,my_id,my_shape,"<=")
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.NotEqualHeaviside):
        id_left,id_right,my_id = self.break_down_binary(symbol)
        my_shape = self.find_broadcastable_shape(id_left,id_right)
        self._intermediate[my_id] = JuliaBitwiseBinaryOperation(id_left,id_right,my_id,my_shape,"<")
        return my_id
    
    @multimethod 
    def _convert_tree_to_intermediate(self,symbol:pybamm.Index):
        assert len(symbol.children)==1
        id_lower = self._convert_tree_to_intermediate(symbol.children[0])
        my_id = symbol.id
        index = symbol.index
        self._intermediate[my_id] = JuliaIndex(id_lower,my_id,index)
        return my_id

    def find_broadcastable_shape(self,id_left,id_right):
        left_shape = self._intermediate[id_left].shape
        right_shape = self._intermediate[id_right].shape
        #check if either is a scalar
        if left_shape == (1,1):
            return right_shape
        elif right_shape==(1,1):
            return left_shape
        elif left_shape==right_shape:
            return left_shape
        elif (left_shape[0]==1) & (right_shape[1]==1):
            return (right_shape[0],left_shape[1])
        elif (right_shape[0]==1) & (left_shape[1]==1):
            return (left_shape[0],right_shape[1])
        elif (right_shape[0]==1) & (right_shape[1]==left_shape[1]):
            return left_shape
        elif (left_shape[0]==1) & (right_shape[1]==left_shape[1]):
            return right_shape
        elif (right_shape[1]==1) & (right_shape[0]==left_shape[0]):
            return left_shape
        elif (left_shape[1]==1) & (right_shape[0]==left_shape[0]):
            return right_shape
        else:
            print("Right type is {}".format(type(self._intermediate[id_right])))
            print("Right Shape is {}".format(right_shape))
            print("Left Shape is {}".format(left_shape))
            raise NotImplementedError("multiplication for the shapes youve requested doesnt work.")
                
    
    #to find the shape, there are a number of elements that should just have the shame shape as their children. This function removes boilerplate by implementing those cases
    def same_shape(self,id_left,id_right):
        left_shape = self._intermediate[id_left].shape
        right_shape = self._intermediate[id_right].shape
        assert left_shape==right_shape
        return left_shape   
    

    #Functions
    #Broadcastable functions have 1 input and 1 output, and the input and output have the same shape. The hard part is that we have to know which is which and pybamm doesn't differentiate between the two. So, we have to do that with an if statement.
    @multimethod
    def _convert_tree_to_intermediate(self,symbol):
            raise NotImplementedError(
            "Conversion to Julia not implemented for a symbol of type '{}'".format(
                type(symbol)
            )
        )

    @multimethod 
    def _convert_tree_to_intermediate(self,symbol:pybamm.Min):
        my_jl_name = symbol.julia_name
        assert len(symbol.children)==1
        my_shape = (1,1)
        input = self._convert_tree_to_intermediate(symbol.children[0])
        my_id = symbol.id
        self._intermediate[my_id] = JuliaMinimumMaximum(my_jl_name,input,my_id,my_shape)
        return my_id
    
    @multimethod 
    def _convert_tree_to_intermediate(self,symbol:pybamm.Max):
        my_jl_name = symbol.julia_name
        assert len(symbol.children)==1
        my_shape = (1,1)
        input = self._convert_tree_to_intermediate(symbol.children[0])
        my_id = symbol.id
        self._intermediate[my_id] = JuliaMinimumMaximum(my_jl_name,input,my_id,my_shape)
        return my_id

    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Function):
        my_jl_name = symbol.julia_name
        assert len(symbol.children)==1
        my_shape = symbol.children[0].shape
        input = self._convert_tree_to_intermediate(symbol.children[0])
        my_id = symbol.id
        self._intermediate[my_id] = JuliaBroadcastableFunction(my_jl_name,input,my_id,my_shape)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Negate):
        my_jl_name = "-"
        assert len(symbol.children)==1
        my_shape = symbol.children[0].shape
        input = self._convert_tree_to_intermediate(symbol.children[0])
        my_id = symbol.id
        self._intermediate[my_id] = JuliaNegation(my_jl_name,input,my_id,my_shape)
        return my_id



    #Constants and Values. There are only 2 of these. They must know their own shapes.
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Matrix):
        assert is_constant_and_can_evaluate(symbol)
        my_id = symbol.id
        value = symbol.evaluate()
        if value.shape==(1,1):
            self._intermediate[my_id] = JuliaScalar(my_id,value)
        else:
            self._intermediate[my_id] = JuliaConstant(my_id,value)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Vector):
        assert is_constant_and_can_evaluate(symbol)
        my_id = symbol.id
        value = symbol.evaluate()
        if value.shape==(1,1):
            self._intermediate[my_id] = JuliaScalar(my_id,value)
        else:
            self._intermediate[my_id] = JuliaConstant(my_id,value)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Scalar):
        assert is_constant_and_can_evaluate(symbol)
        my_id = symbol.id
        value = symbol.evaluate()
        self._intermediate[my_id] = JuliaScalar(my_id,value)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.Time):
        my_id = symbol.id
        self._intermediate[my_id] = JuliaTime(my_id)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.InputParameter):
        my_id = symbol.id
        name = symbol.name
        self._intermediate[my_id] = JuliaInput(my_id,name)
        self._parameter_dict[my_id] = name
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.StateVector):
        my_id = symbol.id
        first_point = symbol.first_point
        last_point = symbol.last_point
        points = (first_point,last_point)
        shape = symbol.shape
        self._intermediate[my_id] = JuliaStateVector(id,points,shape)
        return my_id
    
    @multimethod
    def _convert_tree_to_intermediate(self,symbol:pybamm.StateVectorDot):
        my_id = symbol.id
        first_point = symbol.first_point
        last_point = symbol.last_point
        points = (first_point,last_point)
        shape = symbol.shape
        self._intermediate[my_id] = JuliaStateVectorDot(id,points,shape)
        return my_id
        
    #utilities for code conversion
    def get_variables_for_binary_tree(self,julia_symbol):
        left_input_var_name = self.get_result_variable_name(self._intermediate[julia_symbol.left_input])
        right_input_var_name = self.get_result_variable_name(self._intermediate[julia_symbol.right_input])
        result_var_name = self.get_result_variable_name(julia_symbol)
        return left_input_var_name,right_input_var_name,result_var_name
    
    #convert intermediates to code. Again, all binary trees follow the same pattern so we just define a function to break them down, and then use the MD to find out what code to generate.
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaConcatenation):
        input_var_names = []
        num_cols = julia_symbol.shape[1]
        my_name = self.get_result_variable_name(julia_symbol)

        #assume we don't have tensors. Already asserted that concatenations have to have the same width.
        if num_cols==1:
            right_parenthesis = "]"
            vec=True
        else:
            right_parenthesis = ",:]"
            vec=False
        #do the 0th one outside of the loop to initialize
        child = julia_symbol.children[0]
        child_var = self._intermediate[child]
        child_var_name = self.get_result_variable_name(self._intermediate[child])
        start_row = 1
        if child_var.shape[0] == 0:
            end_row = 1
            code = ""
        elif child_var.shape[0] == 1:
            end_row = 1
            if vec:
                code = "{}[{}{} .=  {}\n".format(my_name,start_row,right_parenthesis,child_var_name)
            else:
                code = "{}[{}{} = (@view {})\n".format(my_name,start_row,right_parenthesis,child_var_name)
        else:
            start_row = 1
            end_row = child_var.shape[0]
            code = "{}[{}:{}{} .= {}\n".format(my_name,start_row,end_row,right_parenthesis,child_var_name)
        
        for child in julia_symbol.children[1:]:
            child_var = self._intermediate[child]
            child_var_name = self.get_result_variable_name(self._intermediate[child])
            if child_var.shape[0] == 0:
                continue
            elif child_var.shape[0] == 1:
                start_row = end_row+1
                end_row = start_row+1
                if vec:
                    code += "{}[{}{} = {}\n".format(my_name,start_row,right_parenthesis,child_var_name)
                else:
                    code += "{}[{}{} .= {}\n".format(my_name,start_row,right_parenthesis,child_var_name)
            else:
                start_row = end_row+1
                end_row = start_row+child_var.shape[0]-1  
                code += "{}[{}:{}{} .= {}\n".format(my_name,start_row,end_row,right_parenthesis,child_var_name)
        
        self._function_string+=code
        return 0
    
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaDomainConcatenation):
        input_var_names = []
        num_cols = julia_symbol.shape[1]
        my_name = self.get_result_variable_name(julia_symbol)

        #assume we don't have tensors. Already asserted that concatenations have to have the same width.
        if num_cols==1:
            right_parenthesis = "]"
            vec=True
        else:
            right_parenthesis = ",:]"
            vec=False
        #do the 0th one outside of the loop to initialize
        end_row = 0
        code = ""
        for i in range(julia_symbol.secondary_dimension_npts):
            for c in range(len(julia_symbol.children)):
                child_var_name = self.get_result_variable_name(self._intermediate[julia_symbol.children[c]])
                this_slice = list(julia_symbol.children_slices[c].values())[0][i]
                start = this_slice.start
                stop = this_slice.stop
                start_row = end_row+1
                end_row = start_row+(stop-start)-1
                code += "{}[{}:{}{} .= (@view {}[{}:{}{})\n".format(my_name,start_row,end_row,right_parenthesis,child_var_name,start+1,stop,right_parenthesis)
        
        self._function_string+=code
        return 0

    
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaMatrixMultiplication):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "mul!({},{},{})\n".format(result_var_name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {} * {}".format(result_var_name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0

    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaAddition):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {} .+ {}\n".format(result_var_name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {} .+ {}".format(result_var_name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0
    
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaSubtraction):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {} .- {}\n".format(result_var_name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {} .- {}".format(result_var_name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0

    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaMultiplication):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {} .* {}\n".format(result_var_name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {} .* {}".format(result_var_name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0 
    
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaDivision):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {} ./ {}\n".format(result_var_name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {} ./ {}".format(result_var_name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0  

    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaBroadcastableFunction):
        result_var_name = self.get_result_variable_name(julia_symbol)
        input_var_name = self.get_result_variable_name(self._intermediate[julia_symbol.input])
        code = "{} .= {}.({})\n".format(result_var_name,julia_symbol.name,input_var_name)
        self._function_string+=code
        return 0

    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaNegation):
        result_var_name = self.get_result_variable_name(julia_symbol)
        input_var_name = self.get_result_variable_name(self._intermediate[julia_symbol.input])
        code = "{} .= -{}\n".format(result_var_name,input_var_name)
        self._function_string+=code
        return 0
    
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaMinimumMaximum):
        result_var_name = self.get_result_variable_name(julia_symbol)
        input_var_name = self.get_result_variable_name(self._intermediate[julia_symbol.input])
        code = "{} .= {}({})\n".format(result_var_name,julia_symbol.name,input_var_name)
        self._function_string+=code
        return 0

    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaMinMax):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {}({},{})\n".format(result_var_name,julia_symbol.name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {}({},{})\n".format(result_var_name,julia_symbol.name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0 

    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaPower):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {}.^{}\n".format(result_var_name,left_input_var_name,right_input_var_name)
        else:
            code = "{} = {}.^{})\n".format(result_var_name,left_input_var_name,right_input_var_name)
        self._function_string+=code
        return 0
    
    @multimethod
    def convert_intermediate_to_code(self,julia_symbol:JuliaBitwiseBinaryOperation):
        left_input_var_name,right_input_var_name,result_var_name = self.get_variables_for_binary_tree(julia_symbol)
        if self._preallocate:
            code = "{} .= {}.{}{}\n".format(result_var_name,left_input_var_name,julia_symbol.operator,right_input_var_name)
        else:
            code = "{} = {}.{}{})\n".format(result_var_name,left_input_var_name,julia_symbol.operator,right_input_var_name)
        self._function_string+=code
        return 0   

    #Cache and Const Creation
    @multimethod
    def create_cache(self,symbol):
        my_id = symbol.output

        cache_shape = self._intermediate[my_id].shape
        cache_id = self._cache_id+1
        self._cache_id = cache_id
        cache_name = "cache_{}".format(cache_id)
        self._cache_dict[symbol.output] = "cs."+cache_name
        if self._cache_type=="standard":
            if cache_shape[1] == 1:
                cache_shape = "({})".format(cache_shape[0])
            self._cache_and_const_string+="{} = zeros{},\n".format(cache_name,cache_shape)
        else:
            raise NotImplementedError("The cache type you've specified has not yet been implemented")
        return 0

    

    def create_const(self,symbol):
        my_id = symbol.id
        const_id = self._const_id+1
        self._const_id = const_id
        const_name = "const_{}".format(const_id)
        self._const_dict[my_id] = "cs."+const_name
        mat_value = symbol.value
        val_line = self.write_const(mat_value)
        const_line = const_name+" = {},\n".format(val_line)
        self._cache_and_const_string+=const_line
        return 0
    
    @multimethod
    def write_const(self,mat_value:numpy.ndarray):
        return mat_value

    @multimethod
    def write_const(self,value:scipy.sparse._csr.csr_matrix):
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
        return val_string
    
    def clear(self):
        self._intermediate = OrderedDict()
        self._function_string = ""
        self._cache_dict = OrderedDict()
        self._cache_and_const_string = ""
        self._const_dict = OrderedDict()
        self._cache_id = 0
        self._const_id = 0
    
    #Just get something working here, so can start actual testing
    def write_function_easy(self,funcname):
        #start with the closure
        self._cache_and_const_string = "begin\ncs = (\n" + self._cache_and_const_string
        self._cache_and_const_string += ")\n"


        top = self._intermediate[next(reversed(self._intermediate))]
        top_var_name = self.get_result_variable_name(top)
        my_shape = top.shape
        if len(self._parameter_dict) != 0:
            parameter_string = ""
            for parameter in self._parameter_dict.items():
                parameter_string+="{},".format(parameter[1])
            parameter_string = parameter_string[0:-1]
            parameter_string += "= p\n"
            self._function_string = parameter_string + self._function_string
        if my_shape[1] != 1:
            self._function_string += "J[:,:] .= {}\nreturn nothing\nend\nend".format(top_var_name)
            self._function_string = "function {}(J, y, p, t)\n".format(funcname) + self._function_string
        elif self._dae_type=="semi-explicit":
            self._function_string+= "dy[:] .= {}\nreturn nothing\nend\nend".format(top_var_name)
            self._function_string = "function {}(dy, y, p, t)\n".format(funcname) + self._function_string
        elif self._dae_type=="implicit":
            self._function_string+="out[:] .= {}\nreturn nothing\nend\nend".format(top_var_name)
            self._function_string = "function {}(out, dy, y, p, t)\n".format(funcname) + self._function_string
        
        

        return 0
        

    #this function will be the top level. 
    def convert_tree_to_intermediate(self,symbol,len_rhs=None):
        if self._dae_type == "implicit":
            assert len_rhs != None
            symbol_minus_dy = []
            end = 0
            for child in symbol.orphans:
                start = end
                end += child.size
                if end <= len_rhs:
                    symbol_minus_dy.append(child - pybamm.StateVectorDot(slice(start, end)))
                else:
                    symbol_minus_dy.append(child)
            symbol = pybamm.numpy_concatenation(*symbol_minus_dy)
        self._convert_tree_to_intermediate(symbol)
        return 0

    #rework this at some point
    def build_julia_code(self,funcname="f"):
        for entry in self._intermediate.values():
            if issubclass(type(entry),JuliaBinaryOperation):
                self.create_cache(entry)
                self.convert_intermediate_to_code(entry)
            elif type(entry) is JuliaConstant:
                self.create_const(entry)
            elif type(entry) is JuliaIndex:
                continue
            elif type(entry) is JuliaStateVector:
                continue
            elif type(entry) is JuliaStateVectorDot:
                continue
            elif type(entry) is JuliaScalar:
                continue
            elif type(entry) is JuliaBroadcastableFunction:
                self.create_cache(entry)
                self.convert_intermediate_to_code(entry)
            elif type(entry) is JuliaNegation:
                self.create_cache(entry)
                self.convert_intermediate_to_code(entry)
            elif type(entry) is JuliaMinimumMaximum:
                self.create_cache(entry)
                self.convert_intermediate_to_code(entry)
            elif type(entry) is JuliaTime:
                continue
            elif type(entry) is JuliaInput:
                continue
            elif issubclass(type(entry),JuliaConcatenation):
                self.create_cache(entry)
                self.convert_intermediate_to_code(entry)
            else:
                raise NotImplementedError("uh oh")
        self.write_function_easy(funcname)
        string = self._cache_and_const_string+self._function_string
        return string
