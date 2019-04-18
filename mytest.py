class Model:
    a = [1,2,3,4]
    b = [7,8,9]

def process(model, inplace=True):
    if inplace:
        new_model = model
    else:
        new_model = Model()
    for idx, val in enumerate(model.a):
        new_model.a[idx] = 2*val
    if not inplace:
        return new_model

model = Model()
print(model.a)
print(process(model, inplace=False).a)
print(model.a)
process(model)
print(model.a)
