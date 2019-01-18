

using Base.Test
using CrackCode.Atoms: AtomsD
using JuLIP: bulk

@testset "Atoms" begin
    @testset "AtomsD" begin
        
        # setup
        atoms = bulk(:Si)
        dict = Dict(:a => 5)

        # generation
        ad = AtomsD(atoms, dict)
        @test ad.atoms == atoms
        @test ad.dict == dict

        # generation, optional arguments
        ad = AtomsD(atoms=atoms, dict=dict)
        @test ad.atoms == atoms
        @test ad.dict == dict

    end
end