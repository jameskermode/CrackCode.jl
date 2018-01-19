
module generic

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

function box_around_point(point, box_width)

    x_c = point[1]
    y_c = point[2]

    box_width_x = box_width[1]
    box_width_y = box_width[2]

    x_min = x_c - box_width_x
    x_max = x_c + box_width_x

    y_min = y_c - box_width_y
    y_max = y_c + box_width_y

    return x_min, x_max, y_min, y_max
end




end
