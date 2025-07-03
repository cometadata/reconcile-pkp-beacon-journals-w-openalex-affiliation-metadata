# sample_pdfs_from_directory

Randomly sample PDF files from a directory.

## Usage

```bash
python sample_pdfs_from_directory.py -i <input_dir> -o <output_dir> -n <num_samples>
```

## Arguments

- `-i, --input_dir`: Source directory containing PDF files (required)
- `-o, --output_dir`: Destination directory for sampled PDFs (required)
- `-n, --num_samples`: Number of PDFs to randomly sample (required)

## Example

```bash
python sample_pdfs_from_directory.py -i /path/to/pdfs -o /path/to/samples -n 10
```

Samples 10 random PDF files from `/path/to/pdfs` and copies them to `/path/to/samples`.