
# include this folder as code
current_directory = pwd()
indices = search(current_directory, "crack-tip-clusters")
path_repo = current_directory[1:indices[length(indices)]]
path_code = joinpath(path_repo, "code", "julia_code", "main.jl")
include(path_code)


using Base.Test
using crack_stuff

verbose=true

code_tests = [
   ("testmanatoms.jl", "manatoms"),
]


println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("  Starting Tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")

@testset "crack_stuff" begin
   for (testfile, testid) in code_tests
      println("=======================")
      println("Testset: $(testid)")
      println("=======================")
      @testset "$(testid)" begin include(testfile); end
   end
end
