#!/usr/bin/env bash
objective_arr=( "logP" "QED" )
method_name="graph_ga"
max_func_calls="5000"

# logging
log_dir="./results/log/molopt"
mkdir -p "$log_dir"

curr_expt_idx=0
for target in "${objective_arr[@]}" ; do

    # Result dir for this target
    res_dir="./results/molopt/${method_name}/${target}"
    mkdir -p "${res_dir}"

    # Run multiple trials
    for trial in {0..2}; do
        output_path="${res_dir}/trial-${trial}.json"
        extra_output="${res_dir}/trial-${trial}-extra.pkl"

        if [[ -f "$output_path" ]]; then
            echo "Results for ${target} trial ${trial} exists. Skipping"

        elif [[ -z "$expt_idx" || "$expt_idx" = "$curr_expt_idx" ]] ; then

            echo "Running ${target} trial ${trial}..."

            PYTHONPATH="$(pwd)/src:$PYTHONPATH" python src/mol_opt/run_${method_name}.py \
                --dataset="./data/dockstring-dataset.tsv" \
                --objective="${target}" \
                --maximize \
                --max_func_calls="${max_func_calls}" \
                --num_cpu=8 \
                \
                --output_path="${output_path}" \
                --extra_output_path="${extra_output}" \
                &> "${log_dir}/${method_name}-${target}_trial${trial}.log"


        fi

        # Increment experiment index after every trial
        curr_expt_idx=$(( curr_expt_idx + 1 ))

    done

done
