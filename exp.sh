export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC=""

export OPENAI_API_KEY=""
export OPENAI_API_BASE=""


model_name="gpt-4o-2024-11-20"
backend="openai"


output_dir="results/gpt-4o"
threads=150
data_name="NoCode-bench/NoCode-bench_Verified"


mkdir -p results
mkdir -p ${output_dir}


python agentless/fl/localize.py --file_level \
                               --output_folder ${output_dir}/file_level \
                               --num_threads ${threads} \
                               --model "${model_name}" \
                               --backend "${backend}" \
                               --dataset "${data_name}" \
                               --skip_existing

python agentless/fl/localize.py --file_level \
                              --irrelevant \
                              --output_folder ${output_dir}/file_level_irrelevant \
                              --num_threads ${threads} \
                              --model "${model_name}" \
                              --backend "${backend}" \
                              --dataset "${data_name}" \
                              --skip_existing

python agentless/fl/retrieve.py --index_type simple \
                              --filter_type given_files \
                              --filter_file ${output_dir}/file_level_irrelevant/loc_outputs.jsonl \
                              --output_folder ${output_dir}/retrievel_embedding \
                              --persist_dir embedding/feature-bench_simple_v1 \
                              --dataset "${data_name}" \
                              --num_threads ${threads}

python agentless/fl/combine.py  --retrieval_loc_file ${output_dir}/retrievel_embedding/retrieve_locs.jsonl \
                              --model_loc_file ${output_dir}/file_level/loc_outputs.jsonl \
                              --top_n 3 \
                              --output_folder ${output_dir}/file_level_combined

python agentless/fl/localize.py --related_level \
                               --output_folder ${output_dir}/related_elements \
                               --top_n 3 \
                               --compress_assign \
                               --compress \
                               --start_file ${output_dir}/file_level_combined/combined_locs.jsonl \
                               --num_threads ${threads} \
                               --model "${model_name}" \
                               --backend "${backend}" \
                               --dataset "${data_name}" \
                               --skip_existing

python agentless/fl/localize.py --fine_grain_line_level \
                               --output_folder ${output_dir}/edit_location_samples \
                               --top_n 3 \
                               --compress \
                               --temperature 0.8 \
                               --num_samples 4 \
                               --start_file ${output_dir}/related_elements/loc_outputs.jsonl \
                               --num_threads ${threads} \
                               --model "${model_name}" \
                               --backend "${backend}" \
                               --dataset "${data_name}" \
                               --skip_existing

python agentless/fl/localize.py --merge \
                              --output_folder ${output_dir}/edit_location_individual \
                              --top_n 3 \
                              --num_samples 4 \
                              --dataset "${data_name}" \
                              --start_file ${output_dir}/edit_location_samples/loc_outputs.jsonl


try_times=("0" "1" "2" "3")
for i in "${try_times[@]}"
do
   python agentless/repair/repair.py --loc_file ${output_dir}/edit_location_individual/loc_merged_${i}-${i}_outputs.jsonl \
                                   --output_folder ${output_dir}/repair_sample_${i} \
                                   --loc_interval \
                                   --top_n=3 \
                                   --context_window=10 \
                                   --max_samples 10  \
                                   --cot \
                                   --diff_format \
                                   --gen_and_process \
                                   --num_threads ${threads} \
                                   --model "${model_name}" \
                                   --dataset "${data_name}" \
                                   --backend "${backend}"
done



python agentless/repair/rerank.py --patch_folder ${output_dir}/repair_sample_1/,${output_dir}/repair_sample_2/,${output_dir}/repair_sample_3/,${output_dir}/repair_sample_0/ \
                                 --num_samples 40 \
                                 --deduplicate \
                                 --output_file all_preds.jsonl

