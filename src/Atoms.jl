module Atoms

    using JuLIP: Atoms

    export AtomsD

    # object to hold JuLIP.Atoms object and assoicated dictionary to hold any data
    # tried to extend JuLIP Atoms object by overloading(?) it, but could not get it to work, might not be possible
    mutable struct AtomsD{Atoms, Dict}
        atoms::Atoms
        dict::Dict
    end
    """
    `AtomsD(;atoms::Atoms=Atoms(), dict::Dict=Dict{Any, Any}())`

    ### Optional Arguements
    - ``atoms::Atoms=Atoms()` : JuLIP Atoms object, default empty Atoms object
    - `dict::Dict=Dict{Any, Any}()` : assoicated dictionary, default empty dictionary

    Group JuLIP.Atoms objects and dictionary into one object
    """
    AtomsD(;atoms=Atoms(), dict=Dict{Any, Any}()) = AtomsD(atoms, dict)

end