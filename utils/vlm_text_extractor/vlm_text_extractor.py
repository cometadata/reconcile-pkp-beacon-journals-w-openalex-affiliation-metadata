#!/usr/bin/env python3

import os
import re
import sys
import json
import glob
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def setup_logger(log_dir=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if log_dir is None:
        log_dir = os.getcwd()
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"extraction_{timestamp}.log")
    
    logger = logging.getLogger('vlm_extractor')
    logger.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


logger = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract text from images using Transformers Vision Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i image.jpg -m nanonets/Nanonets-OCR-s
  
  %(prog)s -i image.jpg -m nanonets/Nanonets-OCR-s --use-vllm --tensor-parallel-size 2
  
  %(prog)s -i /path/to/images/ -m nanonets/Nanonets-OCR-s
  
  %(prog)s -i /base/path -p "*/images/*.jpg" -m nanonets/Nanonets-OCR-s --use-vllm --gpu-memory-utilization 0.8
        """
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input path: single image file, directory containing images, or base directory for pattern matching"
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="HuggingFace model repository (e.g., nanonets/Nanonets-OCR-s)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: auto-generated based on input and model)"
    )

    parser.add_argument(
        "-p", "--pattern",
        help="Subdirectory pattern for recursive image search (e.g., '*/images/*.jpg')"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for text generation (default: 0.0)"
    )

    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume processing by skipping already processed images"
    )
    
    parser.add_argument(
        "--use-fast",
        action="store_true",
        default=True,
        help="Use fast processor (default: True)"
    )
    
    parser.add_argument(
        "--no-use-fast",
        dest="use_fast",
        action="store_false",
        help="Disable fast processor"
    )
    
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use VLLM offline inference instead of local transformers"
    )
    
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )

    return parser.parse_args()


def validate_image(image_path):
    validation_errors = []
    
    try:
        if not os.path.exists(image_path):
            validation_errors.append(f"File not found: {image_path}")
            return validation_errors
        
        file_stats = os.stat(image_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        if file_size_mb > 100:
            validation_errors.append(f"Large file size: {file_size_mb:.2f} MB")
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width * height > 50_000_000:
                    validation_errors.append(f"Very large image dimensions: {width}x{height}")
                
                if img.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF', 'GIF']:
                    validation_errors.append(f"Unusual image format: {img.format}")
                
                if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    validation_errors.append(f"Unusual image mode: {img.mode}")
                    
        except Exception as e:
            validation_errors.append(f"Cannot open image: {str(e)}")
            
    except Exception as e:
        validation_errors.append(f"Validation error: {str(e)}")
    
    return validation_errors


def log_system_info():
    if not logger:
        return
        
    logger.info("="*50)
    logger.info("System Information")
    logger.info("="*50)
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(f"  Total memory: {memory_total:.2f} GB")
            logger.info(f"  Allocated memory: {memory_allocated:.2f} GB")
    else:
        logger.info("CUDA available: False")
    
    logger.info(f"CPU count: {os.cpu_count()}")
    
    logger.info("="*50)


def get_extraction_prompt():
    return """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""


def init_vllm_llm(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, verbose=False):
    if not VLLM_AVAILABLE:
        raise ImportError("VLLM is required for VLLM mode. Install with: pip install vllm")
    
    if verbose:
        print(f"Initializing VLLM offline LLM")
        print(f"Model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
    
    # Try with requested tensor_parallel_size first, then fall back to smaller sizes
    tensor_sizes_to_try = []
    
    # Start with the requested size and work down to divisors
    for size in range(tensor_parallel_size, 0, -1):
        if tensor_parallel_size % size == 0 or size == 1:
            tensor_sizes_to_try.append(size)
    
    # Always ensure 1 is in the list as final fallback
    if 1 not in tensor_sizes_to_try:
        tensor_sizes_to_try.append(1)
    
    last_error = None
    
    for tp_size in tensor_sizes_to_try:
        try:
            if verbose and tp_size != tensor_parallel_size:
                print(f"Falling back to tensor_parallel_size={tp_size}")
            
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=8192  # Adjust based on model requirements
            )
            
            if verbose:
                print(f"VLLM LLM initialized successfully with tensor_parallel_size={tp_size}")
            
            return llm
            
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Failed with tensor_parallel_size={tp_size}: {e}")
            
            # Check if it's a divisibility error and continue trying smaller sizes
            if "is not divisible by" in str(e):
                continue
            else:
                # For other errors, don't continue trying smaller sizes
                break
    
    # If we get here, all attempts failed
    print(f"Error initializing VLLM LLM with all attempted tensor parallel sizes: {last_error}")
    raise last_error


def load_model(model_name, use_fast=True, verbose=False):
    if verbose:
        print(f"Loading model: {model_name}")

    try:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            if verbose:
                print("Using Flash Attention 2")
        except Exception as e:
            if verbose:
                print("Flash Attention 2 not available, using standard attention")
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        processor = AutoProcessor.from_pretrained(model_name, use_fast=use_fast)

        if verbose:
            print(f"Model loaded successfully (use_fast={use_fast})")

        return model, tokenizer, processor
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        sys.exit(1)



def extract_text_from_image_vllm(image_path, llm, max_tokens=2048, temperature=0.0, verbose=False):
    error_details = {}
    
    try:
        start_time = time.time()
        
        if logger:
            logger.info(f"Starting VLLM processing: {image_path}")
            logger.debug(f"Parameters: max_tokens={max_tokens}, temperature={temperature}")
        
        validation_errors = validate_image(image_path)
        if validation_errors:
            if logger:
                logger.warning(f"Image validation warnings for {image_path}: {validation_errors}")
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            file_stats = os.stat(image_path)
            file_size_mb = file_stats.st_size / (1024 * 1024)
            
            if logger:
                logger.debug(f"File size: {file_size_mb:.2f} MB")
            
            # Load image directly for VLLM
            image = Image.open(image_path)
            
            if logger:
                logger.debug(f"Image loaded: {image.size}, mode: {image.mode}")
                
        except Exception as e:
            error_details['phase'] = 'image_loading'
            error_details['error_type'] = type(e).__name__
            raise
        
        try:
            prompt = get_extraction_prompt()
            
            # Create the prompt for VLLM's chat interface
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            if logger:
                logger.debug("Prompt prepared for VLLM")
                
        except Exception as e:
            error_details['phase'] = 'prompt_preparation'
            error_details['error_type'] = type(e).__name__
            raise
        
        try:
            if logger:
                logger.debug("Starting VLLM generation")
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop_token_ids=None
            )
            
            # Generate response using VLLM's chat interface
            outputs = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            # Extract the generated text
            output_text = outputs[0].outputs[0].text
            
            elapsed_time = time.time() - start_time
            
            if logger:
                logger.info(f"Successfully extracted text from {image_path} in {elapsed_time:.2f}s")
                logger.debug(f"Output text length: {len(output_text)}")
            
            return output_text, elapsed_time
            
        except Exception as e:
            error_details['phase'] = 'generation'
            error_details['error_type'] = type(e).__name__
            raise
    
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = f"Error processing {image_path} with VLLM: {str(e)}"
        
        if logger:
            logger.error(error_msg)
            logger.error(f"Error details: {error_details}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        if verbose:
            print(error_msg)
            
        return None, elapsed_time


def extract_text_from_image(image_path, model, tokenizer, processor, max_tokens=2048, temperature=0.0, verbose=False):
    error_details = {}
    
    try:
        start_time = time.time()
        
        if logger:
            logger.info(f"Starting processing: {image_path}")
            logger.debug(f"Parameters: max_tokens={max_tokens}, temperature={temperature}")
        
        validation_errors = validate_image(image_path)
        if validation_errors:
            if logger:
                logger.warning(f"Image validation warnings for {image_path}: {validation_errors}")

        try:
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(image_path)
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                file_stats = os.stat(image_path)
                file_size_mb = file_stats.st_size / (1024 * 1024)
                
                if logger:
                    logger.debug(f"File size: {file_size_mb:.2f} MB")
                
                image = Image.open(image_path)
            
            if logger:
                logger.debug(f"Image format: {image.format}")
                logger.debug(f"Image size: {image.size}")
                logger.debug(f"Image mode: {image.mode}")
                
        except Exception as e:
            error_details['phase'] = 'image_loading'
            error_details['error_type'] = type(e).__name__
            raise

        try:
            prompt = get_extraction_prompt()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ]},
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
                
            if logger:
                logger.debug(f"Chat template applied, text length: {len(text)}")
                
        except Exception as e:
            error_details['phase'] = 'prompt_preparation'
            error_details['error_type'] = type(e).__name__
            raise

        try:
            inputs = processor(text=[text], images=[image],
                               padding=True, return_tensors="pt")
            
            if logger:
                logger.debug(f"Input shapes: {[(k, v.shape) for k, v in inputs.items() if hasattr(v, 'shape')]}")
                
            if torch.cuda.is_available():
                if logger:
                    logger.debug(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            inputs = inputs.to(model.device)
            
            if torch.cuda.is_available():
                if logger:
                    logger.debug(f"GPU memory after input transfer: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
        except Exception as e:
            error_details['phase'] = 'processing'
            error_details['error_type'] = type(e).__name__
            if 'out of memory' in str(e).lower():
                error_details['error_category'] = 'OOM'
            raise

        try:
            with torch.no_grad():
                if logger:
                    logger.debug(f"Starting generation with max_tokens={max_tokens}")
                    
                if temperature == 0.0:
                    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                else:
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature
                    )
                
                if logger:
                    logger.debug(f"Generation complete, output shape: {output_ids.shape}")
                    
        except Exception as e:
            error_details['phase'] = 'generation'
            error_details['error_type'] = type(e).__name__
            if 'out of memory' in str(e).lower():
                error_details['error_category'] = 'OOM'
            raise

        try:
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
                inputs.input_ids, output_ids)]

            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            elapsed_time = time.time() - start_time

            if logger:
                logger.info(f"Successfully extracted text from {image_path} in {elapsed_time:.2f}s")
                logger.debug(f"Output text length: {len(output_text[0])}")

            return output_text[0], elapsed_time
            
        except Exception as e:
            error_details['phase'] = 'decoding'
            error_details['error_type'] = type(e).__name__
            raise

    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = f"Error processing {image_path}: {str(e)}"
        
        if logger:
            logger.error(error_msg)
            logger.error(f"Error details: {error_details}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            if torch.cuda.is_available():
                logger.error(f"GPU memory at error: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                logger.error(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        if verbose:
            print(error_msg)
            
        return None, elapsed_time


def save_output(text, output_path, verbose=False):
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if verbose:
            print(f"Saved output to: {output_path}")

        return True
    except Exception as e:
        print(f"Error saving output to {output_path}: {e}")
        return False


def get_output_filename(image_path, output_dir=None):
    base_name = Path(image_path).stem
    output_name = f"{base_name}.txt"

    if output_dir:
        return os.path.join(output_dir, output_name)
    else:
        return output_name


def process_single_image(image_path, model, tokenizer, processor, output_dir=None, max_tokens=2048, temperature=0.0, verbose=False, use_vllm=False, vllm_llm=None, model_name=None):
    if use_vllm:
        if vllm_llm is None:
            raise ValueError("VLLM LLM is required when use_vllm=True")
        text, elapsed_time = extract_text_from_image_vllm(
            image_path, vllm_llm, max_tokens, temperature, verbose)
    else:
        text, elapsed_time = extract_text_from_image(
            image_path, model, tokenizer, processor, max_tokens, temperature, verbose)

    if text is None:
        return False, elapsed_time, image_path

    output_path = get_output_filename(image_path, output_dir)

    success = save_output(text, output_path, verbose)
    return success, elapsed_time, None if success else image_path


def get_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f"*{ext}"), recursive=False))
        image_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}"), recursive=False))

    return sorted(image_files)


def check_directory_fully_processed(directory, model_name, image_files):
    model_short_name = model_name.split(
        '/')[-1] if '/' in model_name else model_name
    output_dir = os.path.join(directory, f"{model_short_name}_extracted_text")

    if not os.path.exists(output_dir):
        return False, 0, len(image_files)

    processed_count = 0
    for image_path in image_files:
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        if os.path.exists(output_path):
            processed_count += 1

    is_fully_processed = processed_count == len(image_files)
    return is_fully_processed, processed_count, len(image_files)


def process_directory(directory, model, tokenizer, processor, output_dir, model_name, max_tokens=2048, temperature=0.0, verbose=False, resume=False, use_vllm=False, vllm_llm=None):
    image_files = get_image_files(directory)

    if not image_files:
        print(f"No image files found in directory: {directory}")
        return

    if resume:
        is_fully_processed, processed_count, total_count = check_directory_fully_processed(
            directory, model_name, image_files)
        if is_fully_processed:
            print(f"Directory already fully processed: {directory}")
            print(f"  All {total_count} images have been processed. Skipping.")
            return
        elif processed_count > 0:
            print(f"Directory partially processed: {directory}")
            print(f"  {processed_count}/{total_count} images already processed. Resuming...")

    print(f"Found {len(image_files)} images to process")

    if output_dir is None:
        model_short_name = model_name.split(
            '/')[-1] if '/' in model_name else model_name
        output_dir = os.path.join(directory, f"{model_short_name}_extracted_text")

    print(f"Output directory: {output_dir}")

    successful = 0
    failed = 0
    skipped = 0
    total_time = 0
    failed_images = []

    for i, image_path in enumerate(image_files, 1):
        if resume:
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            if os.path.exists(output_path):
                skipped += 1
                print(f"Skipping {i}/{len(image_files)}: {os.path.basename(image_path)} (already processed)")
                continue

        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")

        success, elapsed_time, failed_path = process_single_image(
            image_path, model, tokenizer, processor, output_dir,
            max_tokens, temperature, verbose, use_vllm, vllm_llm, model_name
        )

        if success:
            successful += 1
            total_time += elapsed_time
            print(f"  Completed in {elapsed_time:.2f}s")
        else:
            failed += 1
            if failed_path:
                failed_images.append(failed_path)

    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    if resume and skipped > 0:
        print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(image_files)}")
    if successful > 0:
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per page: {total_time/successful:.2f}s")
    
    if failed_images:
        failed_images_file = os.path.join(output_dir, "failed_images.txt")
        try:
            with open(failed_images_file, 'w') as f:
                for img_path in failed_images:
                    f.write(f"{img_path}\n")
            print(f"\nFailed images list saved to: {failed_images_file}")
        except Exception as e:
            print(f"Error saving failed images list: {e}")


def process_subdirectories(base_path, pattern, model, tokenizer, processor, model_name, max_tokens=2048, temperature=0.0, verbose=False, resume=False, use_vllm=False, vllm_llm=None):
    search_pattern = os.path.join(base_path, pattern)
    matching_files = glob.glob(search_pattern, recursive=True)

    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.gif', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in matching_files if Path(
        f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found matching pattern: {pattern}")
        return

    print(f"Found {len(image_files)} images matching pattern")

    dir_groups = {}
    for image_path in image_files:
        dir_path = os.path.dirname(image_path)
        if dir_path not in dir_groups:
            dir_groups[dir_path] = []
        dir_groups[dir_path].append(image_path)

    total_successful = 0
    total_failed = 0
    total_skipped = 0
    total_dirs_skipped = 0
    total_time = 0

    for dir_path, images in dir_groups.items():
        model_short_name = model_name.split('/')[-1]
        output_dir = os.path.join(dir_path, f"{model_short_name}_extracted_text")

        if resume:
            is_fully_processed, processed_count, total_count = check_directory_fully_processed(
                dir_path, model_name, images)
            if is_fully_processed:
                print(f"\nDirectory already fully processed: {dir_path}")
                print(f"  All {total_count} images have been processed. Skipping.")
                total_skipped += total_count
                total_dirs_skipped += 1
                continue
            elif processed_count > 0:
                print(f"\nDirectory partially processed: {dir_path}")
                print(f"  {processed_count}/{total_count} images already processed. Resuming...")

        print(f"\nProcessing directory: {dir_path}")
        print(f"Output directory: {output_dir}")

        for i, image_path in enumerate(images, 1):
            if resume:
                base_name = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{base_name}.txt")
                if os.path.exists(output_path):
                    total_skipped += 1
                    print(f"  Skipping {i}/{len(images)}: {os.path.basename(image_path)} (already processed)")
                    continue

            print(f"  Processing {i}/{len(images)}: {os.path.basename(image_path)}")

            success, elapsed_time, failed_path = process_single_image(
                image_path, model, tokenizer, processor, output_dir,
                max_tokens, temperature, verbose, use_vllm, vllm_llm, model_name
            )

            if success:
                total_successful += 1
                total_time += elapsed_time
                print(f"    Completed in {elapsed_time:.2f}s")
            else:
                total_failed += 1

    print(f"\nOverall processing complete:")
    print(f"  Successful: {total_successful}")
    if resume and total_skipped > 0:
        print(f"  Skipped: {total_skipped}")
        if total_dirs_skipped > 0:
            print(f"  Fully processed directories skipped: {total_dirs_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Total: {len(image_files)}")
    if total_successful > 0:
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per page: {total_time/total_successful:.2f}s")


def main():
    global logger
    
    args = parse_arguments()
    
    logger, log_file = setup_logger()
    print(f"Logging to: {log_file}")
    
    log_system_info()

    if args.use_vllm:
        print("Using VLLM offline inference mode")
        vllm_llm = init_vllm_llm(
            args.model, 
            args.tensor_parallel_size, 
            args.gpu_memory_utilization, 
            args.verbose
        )
        model = None
        tokenizer = None
        processor = None
    else:
        print("Using local Transformers mode")
        model, tokenizer, processor = load_model(args.model, args.use_fast, args.verbose)
        vllm_llm = None

    input_path = args.input

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        success, elapsed_time, failed_path = process_single_image(
            input_path, model, tokenizer, processor, args.output,
            args.max_tokens, args.temperature, args.verbose,
            args.use_vllm, vllm_llm, args.model
        )

        if success:
            print(f"Text extraction completed successfully in {elapsed_time:.2f}s")
        else:
            print("Text extraction failed")
            if logger:
                logger.error(f"Failed to process: {failed_path}")
            sys.exit(1)

    elif os.path.isdir(input_path) and not args.pattern:
        print(f"Processing directory: {input_path}")

        process_directory(
            input_path, model, tokenizer, processor, args.output, args.model,
            args.max_tokens, args.temperature, args.verbose, args.resume,
            args.use_vllm, vllm_llm
        )

    elif os.path.isdir(input_path) and args.pattern:
        print(f"Processing subdirectories with pattern: {args.pattern}")
        process_subdirectories(
            input_path, args.pattern, model, tokenizer, processor, args.model,
            args.max_tokens, args.temperature, args.verbose, args.resume,
            args.use_vllm, vllm_llm
        )

    else:
        print(f"Error: Invalid input path or pattern: {input_path}")
        sys.exit(1)
    
    if logger:
        logger.info("="*50)
        logger.info("Processing Summary")
        logger.info("="*50)
        logger.info(f"Log file: {log_file}")
        logger.info("Check the log file for detailed error information and debugging traces.")


if __name__ == "__main__":
    main()
