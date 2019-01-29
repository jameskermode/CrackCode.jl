using PyCall

function pip(pkgname)
   pipcmd = `$(PyCall.python) -m pip install --upgrade --user $(pkgname)`
   run(`$(pipcmd)`)
end

println("Installing dependencies `ase`, `matscipy`, `matplotlib` and PyPlot")

try
    Pkg.rm("PyPlot")
catch
end

ENV["PYTHON"]="" # not sure if I need to do this line
Pkg.build("PyCall")
Pkg.add("PyPlot")


try
    @pyimport matplotlib as _matplotlib_
catch
    pip("matplotlib")
end
try
    @pyimport ase as _ase_
catch
    pip("ase")
end

try
    @pyimport matscipy as _matscipy_
catch
    pip("matscipy")
end