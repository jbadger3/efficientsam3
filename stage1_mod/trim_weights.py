#!/usr/bin/env python3
"""
Traverse all PyTorch weight files (.pt, .pth) in a specified directory,
filter the state_dict based on prefix rules, and save the cleaned weights
into the output directory.

Usage example:
    python trim_weights.py --input_dir ./models \
                           --output_dir ./cleaned_models \
                           --prefix "detector.backbone.vision_backbone.trunk." \
                           --preserve_prefix "detector.backbone.vision_backbone.trunk.model." \
                           --preserve_prefix "backbone.bn1"
"""

import os
import argparse
import torch
from pathlib import Path
import warnings


def load_state_dict(filepath):
    """
    Safely load a PyTorch weight file, compatible with different versions
    regarding the 'weights_only' parameter.
    If PyTorch version >= 1.10, use weights_only=True for security;
    otherwise fall back to traditional loading and ignore warnings.
    """
    torch_version = torch.__version__.split('+')[0]  # remove possible local version suffix
    version_parts = list(map(int, torch_version.split('.')))
    print(version_parts)
    if version_parts[0] > 1 or version_parts[1] > 10:
        # PyTorch 1.10+ supports weights_only argument
        try:
            # Try with weights_only=True first
            state_dict = torch.load(filepath, map_location='cpu', weights_only=True)
        except RuntimeError as e:
            # If loading fails because of custom classes, fall back to normal loading (potentially unsafe)
            warnings.warn(f"Loading with weights_only=True failed: {e}. Falling back to normal loading.")
            state_dict = torch.load(filepath, map_location='cpu')
    else:
        # Older PyTorch versions, load directly
        state_dict = torch.load(filepath, map_location='cpu')

    if len(state_dict.keys()) == 1:
        state_dict = [state_dict[x] for x in state_dict.keys()][0]

    return state_dict


def filter_state_dict(state_dict, prefix, preserve_prefixes):
    """
    Remove entries from state_dict whose keys satisfy:
        - key starts with `prefix`
        - AND key does NOT start with any string in `preserve_prefixes`

    Returns state_dict.
    """
    # Determine keys to keep:
    # - keys that do NOT start with prefix are kept unconditionally
    # - keys that start with prefix are kept only if they start with any preserve_prefix
    keys_to_remove = []
    for key in state_dict.keys():
        if key.startswith(prefix):
            # Check if it starts with any of the preserve prefixes
            if preserve_prefixes is not None and any(key.startswith(p) for p in preserve_prefixes):
                continue
            # Otherwise it is dropped
            keys_to_remove.append(key)

    for key in keys_to_remove:
        print(f"Remove key: {key}")
        del state_dict[key]

    return state_dict


def print_keys(state_dict):
    print("keys in state dict:")
    for key in state_dict.keys():
        print(key)


def main():
    parser = argparse.ArgumentParser(description='Clean PyTorch weight files by filtering keys with a given prefix')
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory containing the original weight files')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='output directory')
    parser.add_argument('--prefix', '-p', required=True,
                        default="detector.backbone.vision_backbone.trunk.",
                        help='Prefix to check, e.g. "backbone."')
    parser.add_argument('--preserve_prefix', '-pp', action='append', required=False,
                        help='Full prefixes to preserve (can be specified multiple times), e.g. "backbone.conv1"')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    preserve_prefixes = args.preserve_prefix  # list

    # Supported file extensions
    extensions = ('.pt', '.pth')

    # Traverse input directory
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix in extensions:
            print(f"Processing file: {file_path.name}")
            try:
                # Load weights
                state_dict = load_state_dict(file_path)

                # Filter keys
                filtered_state_dict = filter_state_dict(state_dict, prefix, preserve_prefixes)

                # Print keys
                print_keys(filtered_state_dict)

                # Save to output directory
                output_path = output_dir / file_path.name
                torch.save(filtered_state_dict, output_path)
                print(f"  Saved to: {output_path}")

            except Exception as e:
                print(f"  Error processing file {file_path.name}: {e}")


if __name__ == '__main__':
    main()