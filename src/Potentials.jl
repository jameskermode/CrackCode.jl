# Potentials

module Potentials

using PyPlot
using PyCall
using JuLIP
using JuLIP.Potentials
using ASE

include("ManAtoms.jl")

export idealbrittlesolid, calc_matscipy_ibs, plot_potential


"""
Ideal Brittle Solid Potential

- 0.5*(k-a)^2 is potential energy
- extra 0.5 to match matscipy atom energy vs julia bond energy
- ie ``E_{matscipy} = \sum_{ij} \frac{1}{2} V(r_{ij})`` vs. ``E_{julia} = \sum_{ij} V(r_{ij})``
- r_cut = 1.2 produces reasonable cracks and minimises well
- r_cut = 1.01 matches matscipy_ibs
"""
IdealBrittleSolid(k, a, r_cut=1.2) =
        PairPotential(:(0.5*0.5*$k*(r-$a)^2 - 0.5*0.5*$k*($r_cut-$a)^2), id = "IdealBrittleSolid(k=$k, a=$a)")

idealbrittlesolid(; k=1.0, a=1.0, r_cut=1.2) =
            SplineCutoff(r_cut, r_cut)*IdealBrittleSolid(k, a, r_cut)

idealbrittlesolid_step(; k=1.0, a=1.0, r_cut=1.2) =
            StepFunction(r_cut)*IdealBrittleSolid(k, a, r_cut)

@pyimport matscipy.fracture_mechanics.idealbrittlesolid as ibs
matscipy_ibs = ibs.IdealBrittleSolid()
calc_matscipy_ibs = ASECalculator(matscipy_ibs)


function poisson_ratio_idealbrittlesolid()
    # nu
    return 0.25
end

function youngs_modulus_idealbrittlesolid(k=1.0, a=1.0)
    # E
    # using default values from potentials
    return 5.0*sqrt(3.0)/4.0*k/a
end

function elastic_constants_idealbrittlesolid(E, nu)

    K = E/(3.*(1-2*nu))

    C44 = E/(2.*(1+nu))
    C11 = K+4.*C44/3.
    C12 = K-2.*C44/3.

    return C11, C12, C44
end


function plot_potential(potential)

    # build atoms
    atoms = manatoms.dimer("Si", seperation=0.8, cell_size=30.0)

    calc = potential
    set_calculator!(atoms, calc)
    set_constraint!(atoms, FixedCell(atoms))

    r = linspace(0.4, 2.5, 2000)
    potential_energies = []
    positions = mat(get_positions(atoms))
    for seperation in r
      positions[1,1] = -seperation/2.0
      positions[1,2] = +seperation/2.0
      set_positions!(atoms, positions)
      push!(potential_energies, energy(atoms))
    end

    plot(r, potential_energies, label="$potential")
    xlabel("Seperation, r")
    ylabel("Potential Energy")
    legend()

    return r, potential_energies
end


end
