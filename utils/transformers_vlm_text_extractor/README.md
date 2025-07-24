# VLM Text Extractor

Extract text from images using Vision Language Models with support for both local Transformers and server/VLLM inference.

## Installation

```bash
pip install transformers torch pillow vllm openai
```

## Usage

### Local Transformers Mode (Default)

#### Single Image
```bash
python vlm_text_extractor.py -i image.jpg -m repo/model_name
```

#### Directory of Images
```bash
python vlm_text_extractor.py -i /path/to/images/ -m repo/model_name
```

#### Pattern Matching
```bash
python vlm_text_extractor.py -i /base/path -p "*/images/*.jpg" -m repo/model_name
```

#### Resume Processing
```bash
python vlm_text_extractor.py -i /path/to/images/ -m repo/model_name --resume
```

### VLLM Server Mode

First, start the VLLM server:
```bash
vllm serve repo/model_name
```

Then use the extractor with VLLM mode:

#### Single Image with VLLM
```bash
python vlm_text_extractor.py -i image.jpg -m repo/model_name --use-vllm
```

#### Custom VLLM Server Host/Port
```bash
python vlm_text_extractor.py -i image.jpg -m repo/model_name --use-vllm --vllm-host 192.168.1.100 --vllm-port 8080
```

#### Batch Processing with VLLM
```bash
python vlm_text_extractor.py -i /path/to/images/ -m repo/model_name --use-vllm --resume
```

## Options

### Required Arguments
- `-i, --input`: Input path (single image file, directory, or base directory for pattern matching)
- `-m, --model`: Model name (HuggingFace repository for local mode, or model name for VLLM)

### Optional Arguments
- `-o, --output`: Output directory (default: auto-generated based on input and model)
- `-p, --pattern`: Subdirectory pattern for recursive image search (e.g., '*/images/*.jpg')
- `-r, --resume`: Resume processing by skipping already processed images
- `-v, --verbose`: Enable verbose output

### Generation Parameters
- `--max-tokens`: Maximum tokens to generate (default: 2048)
- `--temperature`: Temperature for text generation (default: 0.0)

### VLLM Mode Options
- `--use-vllm`: Use VLLM server for inference instead of local transformers
- `--vllm-host`: VLLM server host (default: localhost)
- `--vllm-port`: VLLM server port (default: 8000)

### Processor Options
- `--use-fast`: Use fast processor (default: True)
- `--no-use-fast`: Disable fast processor

## Output

Extracted text is saved as `.txt` files in a directory named `{model_name}_extracted_text` within the input directory.


## Examples

### Process a directory with custom output location
```bash
python vlm_text_extractor.py -i /path/to/pdfs/images -m repo/model_name -o /path/to/output
```

### Process all PNG files in nested directories
```bash
python vlm_text_extractor.py -i /documents -p "**/*.png" -m repo/model_name --use-vllm
```