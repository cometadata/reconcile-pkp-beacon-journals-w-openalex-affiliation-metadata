import csv
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Transforms OAI URLs from PKP Beacon input CSV into journal base URLs, "
                    "saving them to an output CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input CSV file. Must contain an 'oai_url' column."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output CSV file where base URLs will be written."
    )
    return parser.parse_args()


def transform_url(oai_url):
    if not oai_url or not isinstance(oai_url, str):
        return ""
    suffixes_to_remove = [
        "/journal/index.php/index/oai",
        "/index.php/index/oai",
        "/oai"
    ]
    transformed_url = oai_url
    for suffix in suffixes_to_remove:
        if transformed_url.endswith(suffix):
            transformed_url = transformed_url[:-len(suffix)]
            return transformed_url

    return transformed_url


def process_input_csv(input_file_path):
    unique_base_urls = set()
    try:
        with open(input_file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            if 'oai_url' not in reader.fieldnames:
                print(f"Error: The input CSV file '{input_file_path}' must contain an 'oai_url' column.")
                return None

            for row_number, row in enumerate(reader, 1):
                oai_url = row.get('oai_url')

                if oai_url and oai_url.strip():
                    base_url = transform_url(oai_url.strip())
                    if base_url:
                        unique_base_urls.add(base_url)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'.")
        return None
    except Exception as e:
        print(f"Error processing CSV file '{input_file_path}': {e}")
        return None

    return sorted(list(unique_base_urls))


def write_output_csv(output_file_path, base_urls):
    if base_urls is None:
        print("Skipping writing output CSV due to previous errors.")
        return
    try:
        with open(output_file_path, mode='w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['base_url'])
            for url in base_urls:
                writer.writerow([url])
        print(f"Successfully wrote {len(base_urls)} unique base URLs to '{output_file_path}'.")
    except IOError:
        print(f"Error: Could not write to output file at '{output_file_path}'. Check permissions or path.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output CSV: {e}")


def main():
    args = parse_arguments()
    print(f"Starting OAI URL processing from input file: {args.input}")
    processed_base_urls = process_input_csv(args.input)
    if processed_base_urls is not None:
        if processed_base_urls:
            write_output_csv(args.output, processed_base_urls)
        else:
            print("No valid OAI URLs found or processed from the input file. Output file will be empty or contain only header.")
            write_output_csv(args.output, [])
    else:
        print("Processing failed. Output file will not be created or may be incomplete.")


if __name__ == "__main__":
    main()
