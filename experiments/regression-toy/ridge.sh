#!/usr/bin/env bash
target_arr=( logP QED )
method_name="ridge"

curr_expt_idx=0
for target in "${target_arr[@]}" ; do

    # Result dir for this target
    res_dir="./results/regression/${method_name}/${target}"
    mkdir -p "${res_dir}"

    # Run multiple trials
    for trial in {0..2}; do
        output_path="${res_dir}/trial-${trial}.json"

        if [[ -f "$output_path" ]]; then
            echo "Results for ${target} trial ${trial} exists. Skipping"

        elif [[ -z "$expt_idx" || "$expt_idx" = "$curr_expt_idx" ]] ; then

            echo "Running ${target} trial ${trial}..."

            PYTHONPATH="$(pwd)/src:$PYTHONPATH" python src/regression/${method_name}.py \
                --data_split="./data/cluster_split.tsv" \
                --dataset="./data/dockstring-dataset-extra-props.tsv" \
                --target="$target" \
                --max_docking_score="inf" \
                \
                --output_path="${output_path}" \

        fi

        # Increment experiment index after every trial
        curr_expt_idx=$(( curr_expt_idx + 1 ))

    done

done
