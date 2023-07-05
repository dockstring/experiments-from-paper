# Script to split up ZINC file into small chunks
zinc_path="data/zinc/zinc20-sorted.csv"
split_size="1000000"
split_dir="data/zinc/split-size-${split_size}"
split_file_name="zinc-split"
tmp_prefix="tmp-"
mkdir -p "$split_dir"

# Do splitting
split -l "$split_size" \
    --numeric-suffixes \
    --additional-suffix=".csv" \
    --suffix-length=9 \
    "$zinc_path" "$split_dir/${tmp_prefix}${split_file_name}"

# Add a header to each file to make the final version
stored_header=""
for tmp_split_file in "$split_dir/${tmp_prefix}${split_file_name}"* ; do

    # What should the new filename be?
    new_name="$(basename ${tmp_split_file})"
    new_name="${split_dir}/${new_name:4}"

    # Check whether it already has the header: if so, do nothing
    first_line="$(head -n 1 ${tmp_split_file})"
    if [[ "$first_line" == *"smiles"* ]] ; then

        # Already has header --> just rename
        mv "$tmp_split_file" "$new_name"
        stored_header="$first_line"
    else

        # Needs header: ensure our header is defined, add the header, remove old file
        if [[ -z "$stored_header" ]]; then
            exit 1  # Shouldn't happen!!
        else
            echo "$stored_header" | cat - "$tmp_split_file" >> "$new_name" && rm "$tmp_split_file"
        fi
    fi
done

echo "SCRIPT FINISHED SUCCESSFULLY."
