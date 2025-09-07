# got-ocr-indic
Implementation and fine-tuning of GOT-OCR 2.0 for Indic language text recognition. The repository provides dataset preprocessing pipelines, modular training and inference scripts, and sample outputs. Designed for high-accuracy OCR on complex, curved, and multi-script text in Hindi and other Indic languages.

## 📥 Dataset

The fine-tuning process uses an Indic OCR dataset containing printed and handwritten text samples.  
For this project, the dataset was preprocessed into the **JSONL format** compatible with GOT-OCR 2.0.

### Structure of JSONL File
Each line in the `.jsonl` file contains:
```json
{
  "image": "path/to/image.jpg",
  "text": "corresponding text label"
}

