
module Generic

#export pad_subset_indices_array, bincount

function pad_subset_indices_array(array_1, array_2)
    # array_1 to pad to length of array_2

    array_1_padded = zeros(array_2)
    for i in 1:length(array_1)
        array_1_padded[array_1[i]] = array_1[i]
    end
    #println("array_1_padded:", array_1_padded)
    return array_1_padded
end


function nonzero_array(array)

    nonzero_indices = find(array .!= 0.0)
    nonzero_array = array[nonzero_indices]

    return nonzero_array

end

function bincount(bins_array, array_to_count)

    counted = zeros(length(bins_array))
    for i in 1:length(array_to_count)
        counted[array_to_count[i]] += 1
    end

    return counted
end





end
