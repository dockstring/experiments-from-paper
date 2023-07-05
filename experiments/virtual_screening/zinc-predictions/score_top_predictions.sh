#!/usr/bin/env bash
# Script to score all top predictions

for method_dir in "./results/virtual-screening/"* ; do
    for protein_dir in "$method_dir/"* ; do
        protein_name="$(basename $protein_dir)"

        # To split, run the SMILES splitting script and add 0* to the string below
        for top_pred_tsv in "${protein_dir}/predictions-trial-"*"-top.tsv" ; do

            # Make sure file actually exists...
            if [[ -f "$top_pred_tsv" ]] ; then
                base_tsv_name="$(basename ${top_pred_tsv} .tsv)"
                output_path="${protein_dir}/${base_tsv_name}-scored.tsv"

                if [[ -f "$output_path" ]] ; then
                    echo "SKIPPING file ${top_pred_tsv} for target ${protein_name}: scores already exist"
                else
                    echo "Scoring file: ${top_pred_tsv} for target ${protein_name}"
                    python "scripts/dock_tsv.py" \
                        --target="${protein_name}" \
                        --input_file="${top_pred_tsv}" \
                        --output_path="$output_path"
                fi
            fi
        done
    done
done
