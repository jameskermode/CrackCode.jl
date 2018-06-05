# Potentials

module Potentials

using PyPlot
using PyCall
using JuLIP
using JuLIP.Potentials
using ASE

    using JuLIP: Atoms, mat, get_positions, set_positions!, set_calculator!, set_constraint!,
                                FixedCell, energy, forces
    using ASE: ASEAtoms

    include("ManAtoms.jl")

    export potential_energy, potential_forces
                idealbrittlesolid, calc_matscipy_ibs, plot_potential

    """
    `potential_energy(atoms::Atoms, potential; r_a = 0.4, r_b = 2.5, points = 100)`

    Assumes atoms object is a dimer

    ### Arguments
    - atoms::Atoms
    - potential
    - r_a : initial seperation distance
    - r_b : final separation distance
    - points : number of points inbetween initial and final distances

    ### Other methods
    `potential_energy(potential; r_a = 0.4, r_b = 2.5, points = 100, cell_size = 30.0)`

    - cell_size : size of the box in which the atoms exist in

    ### Returns
    - r : separation array
    - potential_energies : energies array
    """
    function potential_energy(atoms::Atoms, potential; r_a = 0.4, r_b = 2.5, points = 100)

        calc = potential
        set_calculator!(atoms, calc)
        set_constraint!(atoms, FixedCell(atoms))

        r = linspace(r_a, r_b, points)
        potential_energies = Array{Float64}(points)
        positions = mat(get_positions(atoms))
        for i in 1:points
            seperation = r[i]
            positions[1,1] = -seperation/2.0
            positions[1,2] = +seperation/2.0
            set_positions!(atoms, positions)
            potential_energies[i] = energy(atoms)
        end    

        return r, potential_energies
    end

    function potential_energy(potential; r_a = 0.4, r_b = 2.5, points = 100, cell_size = 30.0)    
        atoms = dimer("H", cell_size = cell_size)
        return potential_energy(atoms, potential; r_a = r_a, r_b = r_b, points = points)
    end

    """
    `potential_forces(atoms::Atoms, potential; r_a = 0.4, r_b = 2.5, points = 100)`

    Assumes atoms object is a dimer. Returns array of the x component of forces on each atom.

    ### Arguments
    - atoms::Atoms
    - potential
    - r_a : initial seperation distance
    - r_b : final separation distance
    - points : number of points inbetween initial and final distances

    ### Other methods
    `potential_forces(potential; r_a = 0.4, r_b = 2.5, points = 100, cell_size = 30.0)`

    - cell_size : size of the box in which the atoms exist in

    ### Returns
    - r : separation array
    - forces_a1 : force_x array on atom 1
    - forces_a2 : force_x array on atom 2
    """
    function potential_forces(atoms::Atoms, potential; r_a = 0.4, r_b = 2.5, points = 100)

        calc = potential
        set_calculator!(atoms, calc)
        set_constraint!(atoms, FixedCell(atoms))
    
        r = linspace(r_a, r_b, points)
        forces_a1 = Array{Float64}(points); forces_a2 = Array{Float64}(points)
        positions = mat(get_positions(atoms))
        for i in 1:points
            seperation = r[i]
            positions[1,1] = -seperation/2.0
            positions[1,2] = +seperation/2.0
            set_positions!(atoms, positions)
            forces_a1[i] = forces(atoms)[1][1]
            forces_a2[i] = forces(atoms)[2][1] 
        end    
    
        return r, forces_a1, forces_a2 
    end

    function potential_forces(potential; r_a = 0.4, r_b = 2.5, points = 100, cell_size = 30.0)
        atoms = dimer("H", cell_size = cell_size)
        return potential_forces(atoms, potential; r_a = r_a, r_b = r_b, points = points)
    end


# old section

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
