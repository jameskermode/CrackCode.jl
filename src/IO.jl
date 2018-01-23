# Input / Output

module IO

import Base.write

using JuLIP: JVecs, AbstractAtoms, get_positions, set_positions!

export write

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
