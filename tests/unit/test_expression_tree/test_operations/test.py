from julia import Main
import numpy as np

f_b = Main.eval(
    """
begin
function f_b(dy,y,a,b)
    dy[1] = a*y[1]
    dy[2] = b*y[2]
    dy
end
end
"""
)
dy = [0, 0]
y = [1, 3]
print(dy)  # returns [0 0]
print(f_b(dy, y, 5, 3))  # returns [5 9]
print(dy)  # returns [0 0] (expected [5 9])

Main.dy = [0, 0]
Main.y = [1, 3]
print(Main.dy)  # returns [0 0]
print(Main.f_b(Main.dy, Main.y, 5, 3))  # returns [5 9]
print(Main.dy)  # returns [0 0] (expected [5 9])
