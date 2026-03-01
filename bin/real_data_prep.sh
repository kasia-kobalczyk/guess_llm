# Real dataset generation
# -----------------------

model_name="meta-llama/Llama-2-7b-hf"

# Download monash
gdown 1sKrpWbD3LvLQ_e5lWgX3wJqT50sTd1aZ
mkdir -p ./data/raw_monash
mv monash.tar.gz ./data
tar -xzf ./data/monash.tar.gz -C ./data/raw_monash
rm ./data/monash.tar.gz

# Generate the dataset
python -m guess_llm.datasets.get_llm_data --config-name=darts_dataset ++model_name=$model_name
python -m guess_llm.datasets.get_llm_data --config-name=monash_dataset ++model_name=$model_name

model_name=$(echo $model_name | awk -F'/' '{print $2}')

# Concatenate the datasets
python -m guess_llm.datasets.concat_real_data --model_name=$model_name

# Create train/test/val splits
python -m guess_llm.datasets.generate_splits \
    --dataset_name="monash_darts_combined" \
    --model_name=$model_name \
    --save_name='random_splits_filtered' \
    --filter_types='relative_median_filter,remove_bitcoin' \
    --split_type='random'

python -m guess_llm.datasets.generate_splits \
    --dataset_name="monash_darts_combined" \
    --model_name=$model_name \
    --save_name='kfold_splits_filtered' \
    --filter_types='relative_median_filter,remove_bitcoin' \
    --split_type='kfold_by_dataset'
