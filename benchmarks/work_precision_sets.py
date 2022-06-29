import numpy as np

def WorkPrecisionSet(prob,
                          abstols, reltols, setups,
                          print_names = False, names = None, appxsol = None,
                          error_estimate = "final",
                          test_dt = None, **kwargs):
    N = len(setups)
    assert names == None or len(setups) == len(names)
    wps = np.empty((N,))
    # if names == None:
    #     names = [str(nameof(type(setup["alg"]))) for setup in setups]
    
    for i in range(0,N):
        if print_names:
            print(names[i])
        
        _abstols = setups[i].get("abstols",abstols) 
        
        _reltols = setups[i].get("reltols",reltols)
        _dts = setups[i].get("dts",None)
        
        filtered_setup = filter(p -> p.first in DiffEqBase.allowedkeywords, setups[i])

        wps[i] = WorkPrecision(prob, setups[i][:alg], _abstols, _reltols, _dts;
                               appxsol = appxsol,
                               error_estimate = error_estimate,
                               name = names[i], kwargs..., filtered_setup...)
    end
    return WorkPrecisionSet(wps, N, abstols, reltols, prob, setups, names, error_estimate,
                            nothing)
end