#!/usr/bin/env bash
target_arr=( KIT PARP1 PGR )
method_name="ridge"

curr_expt_idx=0
for target in "${target_arr[@]}" ; do

    # Result dir for this target
    res_dir="./results/virtual-screening/${method_name}/${target}"
    mkdir -p "${res_dir}"

    # Run multiple trials
    for trial in {0..0}; do
        output_path="${res_dir}/trial-${trial}.json"

        if [[ -f "$output_path" ]]; then
            echo "Results for ${target} trial ${trial} exists. Skipping"

        elif [[ -z "$expt_idx" || "$expt_idx" = "$curr_expt_idx" ]] ; then

            echo "Running ${target} trial ${trial}..."

            PYTHONPATH="$(pwd)/src:$PYTHONPATH" python src/regression/${method_name}.py \
                --dataset="./data/dockstring-dataset.tsv" \
                --target="$target" \
                --max_docking_score="5.0" \
                \
                --output_path="${output_path}" \
                --model_save_dir="${res_dir}/model-${trial}" \

        fi

        # Increment experiment index after every trial
        curr_expt_idx=$(( curr_expt_idx + 1 ))

    done

done
