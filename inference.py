import os
import sys
from PIL import Image
from swift.llm import utils
from swift.llm.train.sft import sft_main  # if needed for inference functions
from transformers import AutoTokenizer, AutoProcessor, VisionEncoderDecoderModel

def load_model(model_path):
    """Load the fine-tuned GOT-OCR 2.0 model"""
    print(f"Loading model from: {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model.processor = processor
    return model, tokenizer

def run_inference(model, tokenizer, image_path):
    """Run OCR inference on a single image"""
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text

if __name__ == "__main__":
    # Example usage:
    # python scripts/inference.py checkpoints/hindi_got_model_3 samples/sample1.png
    if len(sys.argv) != 3:
        print("Usage: python scripts/inference.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Load model
    model, tokenizer = load_model(model_path)

    # Run inference
    prediction = run_inference(model, tokenizer, image_path)
    print(f"Predicted text: {prediction}")
