# Tests for Potentials.jl

using Base.Test
using JuLIP: cutoff
using JuLIP.Potentials: lennardjones

using CrackCode.Potentials: cutoff_adjusted

@testset "Potentials" begin
    @testset "cutoff_adjusted" begin

    # setup
    # Lennard Jones calculator
    r0 = 1.0
    calc = lennardjones(r0=r0, e0=0.01, rcut = (1.0*r0, 1.4*r0)) 

    tol = 1e-6
    r_c = cutoff_adjusted(calc, tol=tol)
    @test r_c <= cutoff(calc)
    end
end