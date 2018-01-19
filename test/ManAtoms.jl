# Tests for ManAtoms.jl

using Base.Test
using CrackCode.ManAtoms: seperation
using JuLIP: mat, AbstractAtoms, positions, set_positions!


# ----- seperation -----
atoms = JuLIP.Atoms("Si2")
new_pos = mat(positions(atoms))
new_pos[:,1] = 1.0
set_positions!(atoms, new_pos)

@testset "ManAtoms" begin
    @testset "Seperation" begin
        @test seperation(atoms, [1, 2]) == sqrt(3)
        @test seperation(atoms, [1, 2]) == sqrt(3)
    end
end



# Old code left for reference and eventually clean up
"""

# include this folder as code
current_directory = pwd()
indices = search(current_directory, "crack-tip-clusters")
path_repo = current_directory[1:indices[length(indices)]]
path_code = joinpath(path_repo, "code", "julia_code", "main.jl")
include(path_code)


using JuLIP
using crack_stuff.manatoms
using Base.Test


# ----- test dimer -----
seperation = 4.0
cell_size = 15.0
atoms_dimer = manatoms.dimer("Cu", seperation=seperation, cell_size=cell_size)

@test atoms_dimer.po["get_chemical_symbols"]() == ["Cu", "Cu"]
@test mat(get_positions(atoms_dimer))[1,1] == -seperation/2
@test mat(get_positions(atoms_dimer))[1,2] == seperation/2
@test get_cell(atoms_dimer)[1,1] == 15.0

# ----- test centre_point -----

# basic atoms
a0 = 5.0 # non default a0 for above construction
atoms = bulk("Al", cubic=true, a=a0)

x, y, z = manatoms.centre_point(atoms)

# not the best test(?) needs bulk() to be working
@test (x, y, z) == (a0/2, a0/2, a0/2)
"""
