# Tests for ManAtoms.jl

using Base.Test
using CrackCode.ManAtoms: AtomsD, write_AtomsD, read_AtomsD, separation, do_w_mod_pos
using JuLIP: mat, vecs, AbstractAtoms, Atoms, positions, set_positions!,
            set_cell!, set_pbc!, set_calculator!, get_positions, forces
using ASE: ASEAtoms

using JuLIP: bulk, get_positions # test separation

# newer test set 
@testset "ManAtoms" begin
    @testset "AtomsD" begin

        # setup
        atoms = bulk(:Si)
        dict = Dict("a" => 5)

        # generation
        ad = AtomsD(atoms, dict)
        @test ad.atoms == atoms
        @test ad.dict == dict

        # generation, optional arguments
        ad = AtomsD(atoms=atoms, dict=dict)
        @test ad.atoms == atoms
        @test ad.dict == dict

        # I/O
        # write
        filename = "test"
        i = write_AtomsD(ad, filename)
        @test i == 0
        # read
        ad_r = read_AtomsD("test")
        @test ad_r.atoms == atoms
        @test ad_r.dict == dict

    end

    @testset "separation" begin
        
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




atoms = bulk(:Si)
set_cell!(atoms, 10.0*eye(3))
set_pbc!(atoms, true)
pos = get_positions(atoms)

r0 = 2.0
lj_spline = JuLIP.Potentials.lennardjones(r0=r0, e0=0.01, rcut = (1.0*r0, 1.2*r0))
set_calculator!(atoms, lj_spline)

# difference set of positions
pos_mod = mat(copy(pos))
pos_mod[1,1] = -sep/4
pos_mod[1,2] = +sep/4
pos_mod = vecs(pos_mod)

@testset "ManAtoms" begin
    # test moved to new section 
    #@testset "separation" begin
        #@test separation(atoms, [1, 2]) == sep
    #end
    @testset "do_w_new_pos" begin

        # test that the positions are different before comparing them
        @test get_positions(atoms) != pos_mod
        @test get_positions(atoms) == pos

        # test a property that we know will be different and the same if assigned
        @test forces(atoms) != do_w_mod_pos(forces, atoms, pos_mod)
        @test forces(atoms) == do_w_mod_pos(forces, atoms, pos)

        # test that the positions were reverted back
        @test get_positions(atoms) == pos
    end
end

# Old code left for reference and eventually clean up
#=

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
separation = 4.0
cell_size = 15.0
atoms_dimer = manatoms.dimer("Cu", separation=separation, cell_size=cell_size)

@test atoms_dimer.po["get_chemical_symbols"]() == ["Cu", "Cu"]
@test mat(get_positions(atoms_dimer))[1,1] == -separation/2
@test mat(get_positions(atoms_dimer))[1,2] == separation/2
@test get_cell(atoms_dimer)[1,1] == 15.0

# ----- test centre_point -----

# basic atoms
a0 = 5.0 # non default a0 for above construction
atoms = bulk("Al", cubic=true, a=a0)

x, y, z = manatoms.centre_point(atoms)

# not the best test(?) needs bulk() to be working
@test (x, y, z) == (a0/2, a0/2, a0/2)
=#
