### Download data
[BBQ Data](https://github.com/nyu-mll/BBQ/tree/main/data): Script to download the data is provided [here](pcgu/download_bbq.sh).

[StereoSet Data](https://github.com/moinnadeem/StereoSet/tree/master/data): Script to prepare the data is provided [here](pcgu/data/prepare_ss.py).

### Run PCGU
```bash
python src/general_similarity_retrain.py \
    -m <base_model_path> \
    -l <learning_rate> \
    -k <num_weight_vectors_to_update> \
    -n <num_epochs> \
    -b <batch_size> \
    --sim-batch-size <update_batch_size> \
    --model-output-dir <path_to_store_updated_model>
```
