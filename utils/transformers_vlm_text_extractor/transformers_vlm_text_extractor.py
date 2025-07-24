#!/usr/bin/env python3

import os
import re
import sys
import json
import glob
import time
import argparse
from pathlib import Path
from datetime import datetime

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract text from images using Transformers Vision Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  %(prog)s -i image.jpg -m nanonets/Nanonets-OCR-s
  
  # Process all images in a directory
  %(prog)s -i /path/to/images/ -m nanonets/Nanonets-OCR-s
  
  # Process images in subdirectories matching a pattern
  %(prog)s -i /base/path -p "*/images/*.jpg" -m nanonets/Nanonets-OCR-s
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

    return parser.parse_args()


def get_extraction_prompt():
    return """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""


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


def extract_text_from_image(image_path, model, tokenizer, processor, max_tokens=2048, temperature=0.0, verbose=False):
    try:
        start_time = time.time()

        if verbose:
            print(f"Processing image: {image_path}")

        if image_path.startswith(('http://', 'https://')):
            image = Image.open(image_path)
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)

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

        inputs = processor(text=[text], images=[image],
                           padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            if temperature == 0.0:
                output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature
                )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
            inputs.input_ids, output_ids)]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Successfully extracted text from {image_path} in {elapsed_time:.2f}s")

        return output_text[0], elapsed_time

    except Exception as e:
        error_msg = f"Error processing {image_path}: {str(e)}"
        if verbose:
            print(error_msg)
        return None, 0


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


def process_single_image(image_path, model, tokenizer, processor, output_dir=None, max_tokens=2048, temperature=0.0, verbose=False):
    text, elapsed_time = extract_text_from_image(
        image_path, model, tokenizer, processor, max_tokens, temperature, verbose)

    if text is None:
        return False, 0

    output_path = get_output_filename(image_path, output_dir)

    success = save_output(text, output_path, verbose)
    return success, elapsed_time


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


def process_directory(directory, model, tokenizer, processor, output_dir, model_name, max_tokens=2048, temperature=0.0, verbose=False, resume=False):
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

    for i, image_path in enumerate(image_files, 1):
        if resume:
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            if os.path.exists(output_path):
                skipped += 1
                print(f"Skipping {i}/{len(image_files)}: {os.path.basename(image_path)} (already processed)")
                continue

        print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")

        success, elapsed_time = process_single_image(
            image_path, model, tokenizer, processor, output_dir,
            max_tokens, temperature, verbose
        )

        if success:
            successful += 1
            total_time += elapsed_time
            print(f"  Completed in {elapsed_time:.2f}s")
        else:
            failed += 1

    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}")
    if resume and skipped > 0:
        print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(image_files)}")
    if successful > 0:
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per page: {total_time/successful:.2f}s")


def process_subdirectories(base_path, pattern, model, tokenizer, processor, model_name, max_tokens=2048, temperature=0.0, verbose=False, resume=False):
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

            success, elapsed_time = process_single_image(
                image_path, model, tokenizer, processor, output_dir,
                max_tokens, temperature, verbose
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
    args = parse_arguments()

    model, tokenizer, processor = load_model(args.model, args.use_fast, args.verbose)

    input_path = args.input

    if os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        success, elapsed_time = process_single_image(
            input_path, model, tokenizer, processor, args.output,
            args.max_tokens, args.temperature, args.verbose
        )

        if success:
            print(f"Text extraction completed successfully in {elapsed_time:.2f}s")
        else:
            print("Text extraction failed")
            sys.exit(1)

    elif os.path.isdir(input_path) and not args.pattern:
        print(f"Processing directory: {input_path}")

        process_directory(
            input_path, model, tokenizer, processor, args.output, args.model,
            args.max_tokens, args.temperature, args.verbose, args.resume
        )

    elif os.path.isdir(input_path) and args.pattern:
        print(f"Processing subdirectories with pattern: {args.pattern}")
        process_subdirectories(
            input_path, args.pattern, model, tokenizer, processor, args.model,
            args.max_tokens, args.temperature, args.verbose, args.resume
        )

    else:
        print(f"Error: Invalid input path or pattern: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
