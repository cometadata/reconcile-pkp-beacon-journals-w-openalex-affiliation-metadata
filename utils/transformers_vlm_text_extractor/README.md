# Transformers VLM Text Extractor

Extract text from images using Vision Language Models (VLMs).


## Installation

```bash
pip install transformers torch pillow
```

## Usage

### Single Image
```bash
python transformers_vlm_text_extractor.py -i image.jpg -m repo/model_name
```

### Directory of Images
```bash
python transformers_vlm_text_extractor.py -i /path/to/images/ -m repo/model_name
```

### Pattern Matching
```bash
python transformers_vlm_text_extractor.py -i /base/path -p "*/images/*.jpg" -m repo/model_name
```

### Resume Processing
```bash
python transformers_vlm_text_extractor.py -i /path/to/images/ -m repo/model_name --resume
```

## Options

- `-i, --input`: Input path (required)
- `-m, --model`: HuggingFace model repository (required)
- `-o, --output`: Output directory (default: auto-generated)
- `-p, --pattern`: Subdirectory pattern for recursive search
- `-r, --resume`: Skip already processed images
- `--max-tokens`: Maximum tokens to generate (default: 2048)
- `--temperature`: Generation temperature (default: 0.0)
- `-v, --verbose`: Enable verbose output

## Output

Extracted text is saved as `.txt` files in a directory named `{model_name}_extracted_text` within the input directory.