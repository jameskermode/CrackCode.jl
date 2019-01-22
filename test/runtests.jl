# Not an offical Julia repository so pull from link
try
    # if it already exists
    Pkg.checkout("CrackCode")
catch
    Pkg.clone("https://github.com/lifelemons/CrackCode.jl")
end

using Base.Test
using CrackCode

include("ManAtoms.jl")
include("Potentials.jl")
