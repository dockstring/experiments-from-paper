# Generic script to split up SMILES file (with headers)
# Required variables: "split_size", "csv_path", "out_dir"
tmp_suffix="-tmp"
mkdir -p "$out_dir"

# Do splitting
split -l "$split_size" \
    --numeric-suffixes \
    --additional-suffix="$tmp_suffix" \
    --suffix-length=9 \
    "$csv_path" "${out_dir}/$(basename $csv_path)"

# Add a header to each file to make the final version
stored_header=""
for tmp_split_file in "$out_dir/"*"${tmp_suffix}" ; do

    # What should the new filename be?
    new_name="${out_dir}/$(basename ${tmp_split_file} ${tmp_suffix})"

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
