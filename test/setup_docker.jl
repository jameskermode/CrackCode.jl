# Setup a docker environment to test this package

# Developed on the `libatomsquip/quip` Docker image, https://hub.docker.com/r/libatomsquip/quip/
# - Jupyter Server
#     - `docker run -it -p 8899:8899 -v ~:/root/$USER libatomsquip/quip`
#     - `http://localhost:8899/`
# - Bash Shell    
#    - `docker run -it -v ~:/root/$USER libatomsquip/quip bash`

# Usage
# in julia run:
# include("path.../CrackCode.jl/test/setup_docker.jl")
# push!(LOAD_PATH, path_to_folder_containing_local_module)
# using CrackCode

module setup_docker

    # get function from module not already installed for use in later function
    function initialise_use_package(use_package)
        try Pkg.installed("SciScriptTools")
        catch Pkg.clone("https://github.com/lifelemons/SciScriptTools.jl") end
        @eval using $(Symbol("SciScriptTools"))
        use_package = SciScriptTools.Import.use_package
        return use_package
    end

    function sort_require_modules()

        # Logging and SciScriptTools already obtained
        use_package("ASE", repo = "https://github.com/libAtoms/ASE.jl")
        use_package("LsqFit")
        use_package("StaticArrays")
        use_package("JuLIP")
        use_package("Optim")
        use_package("PyPlot")

    end


    # Setup Script

    # define function in global scope
    use_package = nothing

    # setup Logging package and level
    # to get info levels of logs from function use_package()
    Pkg.add("Logging") # https://github.com/kmsquire/Logging.jl
    using Logging
    Logging.configure(level=INFO)
    use_package = initialise_use_package(use_package) # fill varibale with function from module
    
    # get the rest of the modules
    sort_require_modules() # install (and 'use') modules for docker image

    @printf "Now run:\npush!(LOAD_PATH, path_to_folder_containing_local_module)\n"
    @printf "Then:\nusing CrackCode\n"
end