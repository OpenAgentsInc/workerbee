import argparse

from ai_worker.util import quantize_gguf, GGML_INVERSE_MAP

import argparse

# List of quantization levels
quantization_levels = list(GGML_INVERSE_MAP.keys())

def parse_arguments(argv=None):
    """
    Parse command-line arguments to quantize a gguf (llm model) file.
    """
    parser = argparse.ArgumentParser(description="Quantize a gguf (llm model) file")

    # Input file argument
    parser.add_argument("input_file", type=str, help="Path to the input gguf (llm model) file")

    # Quantization level argument with custom type and choices
    parser.add_argument(
        "quantization_level",
        type=str.lower,  # Convert the input to lowercase to make it case-insensitive
        choices=quantization_levels,  # Use quantization levels as choices
        help="Quantization level (case-insensitive)",
    )

    return parser.parse_args()



def main(argv=None):
    args = parse_arguments(argv)
    
    quantize_gguf(args.input_file, args.quantization_level)


if __name__ == "__main__":
    main()
