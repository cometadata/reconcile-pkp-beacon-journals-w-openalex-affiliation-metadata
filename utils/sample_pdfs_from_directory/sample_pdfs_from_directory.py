import os
import sys
import shutil
import random
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sample a specified number of PDF files from a directory and copy them to an output directory.")
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Directory containing the PDF files to sample from.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Directory where the sampled PDF files will be copied.")
    parser.add_argument("-n", "--num_samples", required=True,
                        type=int, help="The number of PDF files to sample.")
    return parser.parse_args()


def get_pdf_files(directory_path):
    pdf_files = []
    try:
        for item_name in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item_name)
            if os.path.isfile(item_path) and item_name.lower().endswith(".pdf"):
                pdf_files.append(item_name)
    except OSError as e:
        raise IOError(f"Could not read directory contents for '{directory_path}': {e}")
    return pdf_files


def sample_pdf_files(pdf_files_list, num_to_sample):
    if num_to_sample > len(pdf_files_list):
        raise ValueError(f"Cannot sample {num_to_sample} PDFs: only {len(pdf_files_list)} available in the source.")
    return random.sample(pdf_files_list, num_to_sample)


def copy_sampled_files(files_to_copy, source_dir, destination_dir):
    try:
        if os.path.exists(destination_dir):
            if not os.path.isdir(destination_dir):
                raise IOError(f"Output path '{destination_dir}' exists but is not a directory.")
        else:
            os.makedirs(destination_dir)
            print(f"Created output directory: '{destination_dir}'")
    except OSError as e:
        raise IOError(f"Error with output directory '{destination_dir}': {e}")

    successful_copies = 0
    for pdf_filename in files_to_copy:
        source_file_path = os.path.join(source_dir, pdf_filename)
        destination_file_path = os.path.join(destination_dir, pdf_filename)

        if not os.path.isfile(source_file_path):
            print(f"Warning: Source file '{source_file_path}' not found or is not a file (it may have been moved or deleted after scanning). Skipping.")
            continue

        try:
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied '{source_file_path}' to '{destination_file_path}'")
            successful_copies += 1
        except Exception as e:
            print(f"Warning: Failed to copy '{source_file_path}' to '{destination_file_path}'. Error: {e}")

    return successful_copies


def main():
    args = parse_arguments()

    input_directory = os.path.abspath(args.input_dir)
    output_directory = os.path.abspath(args.output_dir)
    num_to_sample = args.num_samples

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found or is not a directory.")
        sys.exit(1)

    if num_to_sample <= 0:
        print(f"Error: Number of samples must be a positive integer. Provided: {num_to_sample}")
        sys.exit(1)

    try:
        pdf_file_names = get_pdf_files(input_directory)
        if not pdf_file_names:
            print(f"No PDF files found in '{input_directory}'. Exiting.")
            sys.exit(0)

        display_limit = 5
        found_files_str = ", ".join(pdf_file_names[:display_limit])
        if len(pdf_file_names) > display_limit:
            found_files_str += f", and {len(pdf_file_names) - display_limit} more"
        print(f"Found {len(pdf_file_names)} PDF(s) in '{input_directory}': {found_files_str}.")

        sampled_files = sample_pdf_files(pdf_file_names, num_to_sample)

        selected_files_str = ", ".join(sampled_files[:display_limit])
        if len(sampled_files) > display_limit:
            selected_files_str += f", and {len(sampled_files) - display_limit} more"
        print(f"Selected {len(sampled_files)} PDF(s) for sampling: {selected_files_str}.")

        if input_directory == output_directory:
            print(f"Input and output directories are the same ('{input_directory}').")
            print("The sampled files are already in the target directory. No physical copy operation will be performed.")
            print("Sampled files (already in target directory):")
            for fname in sampled_files:
                print(f"  - {os.path.join(output_directory, fname)}")
        else:
            num_copied = copy_sampled_files(
                sampled_files, input_directory, output_directory)
            if num_copied == len(sampled_files):
                print(f"Successfully copied all {num_copied} selected PDF(s) to '{output_directory}'.")
            elif num_copied > 0:
                print(f"Partially successful: Copied {num_copied} out of {len(sampled_files)} selected PDF(s) to '{output_directory}'. Check warnings above.")
            else:
                print(f"Operation failed: No files were copied to '{output_directory}'. Check warnings above.")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        sys.exit(1)
    except IOError as ioe:
        print(f"File System Error: {ioe}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
