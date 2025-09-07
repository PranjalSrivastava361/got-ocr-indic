from swift.llm.train.sft import sft_main
from swift.llm import utils
import types
import os
import sys

# Monkey patch to fix gradient checkpointing error
def safe_gradient_checkpointing(model, enable):
    if not enable:
        return
    model_tower = getattr(model.model, "vision_tower_high", None)
    if model_tower is not None:
        model_tower.supports_gradient_checkpointing = True
    else:
        print("vision_tower_high is None. Skipping checkpoint support.")

utils.dynamic_gradient_checkpointing = safe_gradient_checkpointing

# Set visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Simulate command-line arguments
sys.argv = [
    "train_gotocr.py",
    "--model_type", "got_ocr2",
    "--model", "stepfun-ai/GOT-OCR2_0",
    "--dataset", "/kaggle/working/output_data.json",
    "--output_dir", "/kaggle/working/hindi_got_model_3",
    "--num_train_epochs", "1",
    "--max_steps", "4000",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "2e-5",
    "--lora_rank", "8",
    "--lora_alpha", "32",
    "--lora_dropout", "0.05",
    "--save_strategy", "steps",
    "--save_steps", "500",
    "--save_only_model", "False"
    # don't pass vit_gradient_checkpointing
]

# Start training
sft_main()
