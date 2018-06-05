# Tests for ManAtoms.jl

using Base.Test
using CrackCode.ManAtoms: seperation, do_w_mod_pos
using JuLIP: mat, vecs, AbstractAtoms, Atoms, positions, set_positions!,
            set_cell!, set_pbc!, set_calculator!, get_positions, forces


using JuLIP: bulk, get_positions # test seperation

# newer test set 
@testset "ManAtoms" begin
    @testset "seperation" begin
        
        # setup
        atoms = bulk(:Si, cubic=true)
        pos0 = get_positions(atoms)
        pairs = [(1,2), (3,4), (2,3)]
        pair = pairs[1]
        b0 = norm(pos0[pair[2]] - pos0[pair[1]])
        tol = 1e-10

        sep = separation(pos0, pair[1], pair[2])
        @test abs(sep - b0) <= 0.0 + tol
        
        sep = separation(pos0, pair)
        @test abs(sep - b0) <= 0.0 + tol
        
        sep = separation(atoms, pair[1], pair[2])
        @test abs(sep - b0) <= 0.0 + tol
        
        sep = separation(atoms, pair)
        @test abs(sep - b0) <= 0.0 + tol
        
        sep = separation(atoms, pairs)
        @test maximum(abs.(sep - b0)) .<= 0.0 + tol
    end
end




atoms = Atoms("Si2")
set_cell!(atoms, 10.0*eye(3))
set_pbc!(atoms, true)
pos = mat(get_positions(atoms))
sep = 0.8
pos[1,1] += -sep/2
pos[1,2] += +sep/2
pos = vecs(pos)
set_positions!(atoms, pos)

r0 = 1.0
lj_spline = JuLIP.Potentials.lennardjones(r0=r0, e0=0.01, rcut = (1.0*r0, 1.2*r0))
set_calculator!(atoms, lj_spline)

# difference set of positions
pos_mod = mat(copy(pos))
pos_mod[1,1] = -sep/4
pos_mod[1,2] = +sep/4
pos_mod = vecs(pos_mod)

@testset "ManAtoms" begin
    # test moved to new section 
    #@testset "seperation" begin
        #@test seperation(atoms, [1, 2]) == sep
    #end
    @testset "do_w_new_pos" begin

        # test that the positions are different before comparing them
        @test get_positions(atoms) != pos_mod
        @test get_positions(atoms) == pos

        # test a property that we know will be different and the same if assigned
        @test forces(atoms) != do_w_mod_pos(forces, atoms, pos_mod)
        @test forces(atoms) == do_w_mod_pos(forces, atoms, pos)

        # Hard coded numbers for LJ potential above - probably not a great idea!
        # If these fail, something with the potential has likely changed
        @test norm(maximum(do_w_mod_pos(forces, atoms, pos_mod))) < 17808.15124511724 * (1 + 1e-15)
        @test norm(maximum(do_w_mod_pos(forces, atoms, pos_mod))) > 17808.15124511724 * (1 - 1e-15)
        @test norm(maximum(forces(atoms))) < 1.6105826944112733 * (1 + 1e-15)
        @test norm(maximum(forces(atoms))) > 1.6105826944112733 * (1 - 1e-15)


        # test that the positions were reverted back
        @test get_positions(atoms) == pos
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
