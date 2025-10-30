import sys

# Check if the filename argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

# Get the filename from the command-line arguments
filename = sys.argv[1]

try:
    # Open the file and read all lines
    with open(filename, "r") as file:
        lines = file.readlines()

    # Initialize variables to track the minimum value and its line index
    min_value = float('inf')  # Start with infinity
    min_index = -1            # Placeholder for the line index

    # Iterate through the lines to find the minimum value
    for index, line in enumerate(lines):
        try:
            # Convert the line to a float or integer
            value = float(line.strip())
            # Update minimum value and line index if a smaller value is found
            if value < min_value:
                min_value = value
                min_index = index
        except ValueError:
            print(f"Skipping invalid line at index {index}: {line.strip()}")

    # Print the result
    if min_index != -1:
        print(f"Minimum value: {min_value} found at line {min_index + 1}")
    else:
        print("No valid numbers found in the file.")

except FileNotFoundError:
    print(f"File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

