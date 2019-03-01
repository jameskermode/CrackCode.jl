# Input / Output

module IO

    import Base.write

    using JuLIP: JVecs, JVecsF, AbstractAtoms, Atoms, get_positions, set_positions!
    using SciScriptTools.IO: find_files, write_json
    using ASE: ASEAtoms, read_xyz
    import ASE: write_xyz
    using JSON: parsefile, print
    using Logging: info

    export write, read_xyzjson, write_xyzjson, read_pos, write_pos, write_xyz

    """
    Read in both .xyz and associated json files

    ### Arguments
    filename::AbstractString : filename without file extensions
    """
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

    """
    Write both .xyz and associated json files

    ### Arguments
    filename::AbstractString : filename without file extensions
    atoms::Atoms : JuLIP Atoms object
    atoms_dict::Dict : associated dictionary
    """
    function write_xyzjson(filename::AbstractString, atoms::Atoms, atoms_dict::Dict)

        info(@sprintf("writing: %s .xyz and .json", filename))

        write_xyz(string(filename, ".xyz"), ASEAtoms(atoms))

        json_file = open(string(filename, ".json"), "w")
        print(json_file, atoms_dict)
        close(json_file)

        return 0
    end

    """
    `read_pos(filename::AbstractString)`

    Read atoms positions stored in a .json file
    Can be a single set or array of positions
    """
    function read_pos(filename::AbstractString)

        d = parsefile(filename)
        k = collect(keys(d))

        pos = nothing
        f = 0 # fl
        for i in 1:length(k)
            if f == 0
                try
                pos = JVecsF(d[k[i]])
                f = 1
                catch end
            end
            if f == 0
                try
                pos = JVecsF.(d[k[i]])
                f = 1
                catch end
            end
            if f == 0
                error("Not a valid set of positions or array of positions")
                return 1
            end

        end
        return pos
    end

    """
    `write_pos(filename::AbstractString, pos)`

    Write positions, `pos`, to a .json file

    ### Arguments
    - `filename::AbstractString
    - `pos`::`JVecsF` or `Array{JVecsF} : positions

    """
    function write_pos(filename::AbstractString, pos)
        d = Dict("pos" => pos)
        write_json(filename, d)
        return 0
    end

    """
    `write_xyz(filename::AbstractString, atoms::Atoms, path::Array{JVecsF})`

    Write a path to a file .xyz, using `JuLIP Atoms` and `Array{JVecsF}`

    ### Arguments
    - `filename::AbstractString`
    - `atoms::Atoms` : JuLIP Atoms object
    - `path::Array{JVecsF}` : array of atom positions
    """
    function write_xyz(filename::AbstractString, atoms::Atoms, path::Array{JVecsF})
        mode = 'w'
        for i in 1:length(path)
            if i > 1 mode = 'a' end
            set_positions!(atoms, path[i])
            write_xyz(filename, ASEAtoms(atoms), mode)
        end
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
