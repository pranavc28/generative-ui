#!/usr/bin/env python3
"""
Script to extract a specified number of results from eval_results.jsonl
and write them to a new file.
"""

import json
import argparse
from pathlib import Path


def extract_results(input_file, output_file, num_results):
    """
    Extract a specified number of results from a JSONL file.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        num_results (int): Number of results to extract
    """
    try:
        results_extracted = 0
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                if results_extracted >= num_results:
                    break
                
                try:
                    # Parse the JSON line
                    result = json.loads(line.strip())
                    
                    # Write to output file
                    outfile.write(json.dumps(result) + '\n')
                    results_extracted += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON parsing error: {e}")
                    continue
        
        print(f"Successfully extracted {results_extracted} results from {input_file}")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return


def main():
    parser = argparse.ArgumentParser(
        description='Extract a specified number of results from eval_results.jsonl'
    )
    
    parser.add_argument(
        '-n', '--num-results',
        type=int,
        required=True,
        help='Number of results to extract'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='outputs/eval_results.jsonl',
        help='Input JSONL file path (default: outputs/eval_results.jsonl)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='outputs/extracted_results.jsonl',
        help='Output JSONL file path (default: outputs/extracted_results.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths if needed
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate num_results
    if args.num_results <= 0:
        print("Error: Number of results must be positive.")
        return
    
    print(f"Extracting {args.num_results} results...")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print()
    
    extract_results(str(input_path), str(output_path), args.num_results)


if __name__ == "__main__":
    main()

