#!/usr/bin/env bash
# Script to get all top predictions
n_top=5000

for method_dir in "./results/virtual-screening/"* ; do
    for protein_dir in "$method_dir/"* ; do
        for pred_dir in "${protein_dir}/predictions-trial-"* ; do
            if [[ -d "$pred_dir" ]] ; then
                python "scripts/virtual_screening_top_pred.py" \
                    --input_files "${pred_dir}/"*.*sv \
                    --n_top="$n_top" \
                    --output_path="${pred_dir}-top.tsv" \

            fi
        done
    done
done
