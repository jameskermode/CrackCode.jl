# Input / Output

module IO

    import Base.write

    using JuLIP: JVecs, AbstractAtoms, Atoms, get_positions, set_positions!
    using SciScriptTools.IO: find_files
    using SciScriptTools.Conversion: dict_convert_keys
    using ASE: ASEAtoms, read_xyz, write_xyz
    using JSON: parsefile, print
    using Logging: info

    export write, read_xyzjson, write_xyzjson

    "Read in both .xyz and associated json files"
    function read_xyzjson(filename::AbstractString)

        # look for the json file as there is only one json file for each system R
        file_str = find_files(filename, suffix="json")
        [file_str[i] = splitext(file_str[i])[1] for i in 1:length(file_str)]
        info(@sprintf("reading in: %s .xyz and .json", file_str))

        atoms_a = Array{Atoms}(length(file_str)) # array of atoms objects
        atoms_dict_a = Array{Dict}(length(file_str)) # array of dict objects
        for i in 1:length(file_str)
            atoms_a[i] = Atoms(read_xyz(string(file_str[i], ".xyz")))
            atoms_dict_a[i] = parsefile(string(file_str[i], ".json"))
        end

        # in dictionary convert string to symbols
        for i in 1:length(atoms_dict_a)
            atoms_dict_a[i] = dict_convert_keys(atoms_dict_a[i])
        end

        if length(atoms_a) == 1
            return atoms_a[1], atoms_dict_a[1] end
        return atoms_a, atoms_dict_a
    end

    function write_xyzjson(filename::AbstractString, atoms::Atoms, atoms_dict::Dict)

        info(@sprintf("writing: %s .xyz and .json", filename))

        write_xyz(string(filename, ".xyz"), ASEAtoms(atoms))

        json_file = open(string(filename, ".json"), "w")
        print(json_file, atoms_dict)
        close(json_file)

        return 0
    end

# This function seems silly, should find a better way to pass new set of positions
# or maybe just keep this as a script function since its fairly specific and not
# very useful in other cases?!
"""
`write(filename::AbstractString, atoms::AbstractAtoms,
                    pos::JVecs{Float64}, mode = :write)`

Similar to `JuLIP.write()`, can now give set positions
(rather than whatever is within the atoms object)

### Arguements
- `filename::AbstractString`
- `atoms::AbstractAtoms`: atoms object
- `pos::JVecs{Float64}`: new positions
- `mode = :write`: or use :append
"""
function write(filename::AbstractString, atoms::AbstractAtoms,
                    pos::JVecs{Float64}, mode = :write)
    pos_original = get_positions(atoms)
    set_positions!(atoms, pos)
    write(filename, atoms, mode)
    set_positions!(atoms, pos_original)
end


end
