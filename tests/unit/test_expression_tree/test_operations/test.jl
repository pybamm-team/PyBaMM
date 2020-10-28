function f_b(dy, y, a, b)
    dy[1] = a * y[1]
    dy[2] = b * y[2]
    dy
end
dy = [0, 0]
y = [1, 3]
println(dy)  # returns [0 0]
println(f_b(dy, y, 5, 3))  # returns [5 9]
println(dy)  # returns [5 9]