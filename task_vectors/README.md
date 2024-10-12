### Fine-tune model: Make model more biased
```bash
python train_model.py \
    --model_name <base_model_path> \
    --output_path <path_to_store_finetuned_model> \
    --pem "lora_adapter"
```

### Apply task vector negation
```bash
python negation.py \
    --pretrained_model_path <base_model_path> \
    --finetuned_model_path <finetuned_model_path> \
    --scaling_coef <scaling_coef> \
    --model_save_path <path_to_save_final_model>
```
