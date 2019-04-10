def fun(a, b={}):
    if b:
        print(b)
        if a in b:
            return b[a]
        else:
            b[a] = fun(a/2, b)*2
            return b[a]


b = {1: 100}
print(fun(16,b), b)

def t(length, result = [], d = {}):
    if length == 10:
        return
    else:
        result.append(length)
        d[length] = length + 1
        print(d)
        t(length + 1)

    return (result, d)

x, y = t(0)
print(x,y)
