# Input / Output

module IO

    import Base.write

    using JuLIP: JVecs, AbstractAtoms, Atoms, get_positions, set_positions!
    using SciScriptTools.IO: find_files
    using ASE: ASEAtoms, read_xyz, write_xyz
    using JSON: parsefile, print
    using Logging: info

    export write, read_xyzjson, write_xyzjson

    "Read in both .xyz and associated json files"
    function read_xyzjson(filename::AbstractString)

        # find and extract just the filenames
        function get_filenames(filename::AbstractString, format::AbstractString)
            list_f = find_files(filename, suffix=format)
            list_n = [splitext(list_f[i])[1] for i in 1:length(list_f)]
            return list_n
        end

        # look for the json files and xyz files, find pairs with the same name
        files_jsons = get_filenames(filename, ".json")
        files_xyzs  = get_filenames(filename, ".xyz")
        file_str = intersect(files_jsons, files_xyzs)
        info(@sprintf("reading in: %s .xyz and .json", file_str))

        atoms_a = Array{Atoms}(length(file_str)) # array of atoms objects
        atoms_dict_a = Array{Dict}(length(file_str)) # array of dict objects
        for i in 1:length(file_str)
            atoms_a[i] = Atoms(read_xyz(string(file_str[i], ".xyz")))
            atoms_dict_a[i] = parsefile(string(file_str[i], ".json"))
        end

        if length(atoms_a) == 1 return atoms_a[1], atoms_dict_a[1] end
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
