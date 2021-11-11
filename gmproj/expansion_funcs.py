# Older versions of the Expansion functions which don't work in PyBaMM but work outside of it
def graphite_volume_change(sto):
    a = 0.50
    b = 0.24
    c = 0.18 
    d = 0.12
    t_change = 0
    if sto < d:
        t_change = 2.4060/d*sto
    elif d <= sto and sto < c:
        t_change = -(2.4060-3.3568)/(c-d)*(sto-d)+2.4060
    elif c <= sto and sto < b:
        t_change = -(3.3568-4.3668)/(b-c)*(sto-c)+3.3568
    elif b <= sto and sto < a:
        t_change = -(4.3668-5.583)/(a-b)*(sto-b)+4.3668
    elif a <= sto:
        t_change = -(5.583-13.0635)/(1-a)*(sto-a)+5.583
    t_change = t_change/100
    return t_change

def graphite_volume_change(sto):
    a = 0.50
    b = 0.24
    c = 0.18 
    d = 0.12
    t_change = (
        (0 <= sto and sto < d)*(2.4060/d*sto)+
        (d <= sto and sto < c)*(-(2.4060-3.3568)/(c-d)*(sto-d)+2.4060)+
        (c <= sto and sto < b)*(-(3.3568-4.3668)/(b-c)*(sto-c)+3.3568)+
        (b <= sto and sto < a)*(-(4.3668-5.5830)/(a-b)*(sto-b)+4.3668)+
        (a <= sto and sto <= 1)*(-(5.583-13.0635)/(1-a)*(sto-a)+5.583)
    )
    t_change = t_change/100
    return t_change