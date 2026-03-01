# Synthetic dataset generation
# -----------------------

model_name="meta-llama/Llama-2-7b-hf"

for scale in 1.0 10.0 1000.0 10000.0
do  
    # Generate the dataset
    python -m guess_llm.datasets.get_llm_data --config-name=multi_func_${scale}_dataset ++model_name=$model_name
done

model_name=$(echo $model_name | awk -F'/' '{print $NF}')

# Combine all the datasets and generate the train/test/val splits scale-controlled datasets
python -m guess_llm.datasets.concat_synt_data --model_name=$model_name

# Generate train/test/val splits for the combined dataset
python -m guess_llm.datasets.generate_splits \
    --model_name=$model_name \
    --dataset_name="multi_func_combined" \
    --save_name='strong_filter'
