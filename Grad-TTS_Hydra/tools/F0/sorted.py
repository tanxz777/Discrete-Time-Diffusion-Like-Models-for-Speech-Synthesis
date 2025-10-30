#for given directory 'dir', sort all its first level subdiectory and sort them according to numeric part in its directory name

import os
import argparse

def numeric_sort_key(subdir):
    numeric_part = ''.join(filter(str.isdigit, subdir))
    return int(numeric_part)

def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="sort the sub-dir name in a dir.")
    parser.add_argument(
        '-d', '--dir',
        type=str,
        help="Path of directory contains the sub-dirs to be sorted",
    )
    return parser


def main(): 
    """Run MCD calculation in parallel."""
    args = get_parser().parse_args()
    dir = args.dir

    subdirectories = [subdir for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]
    
    sorted_subdirectories = sorted(subdirectories, key=numeric_sort_key)
    sorted_directories = [os.path.join(dir, subdirectory) for subdirectory in sorted_subdirectories]
    directories = " ".join(sorted_directories)
    print(directories)

if __name__ == "__main__":
    main()
