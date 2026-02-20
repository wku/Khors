import sys
import pathlib

def add_comment_to_file(filepath: str, comment: str, line_number: int = 1):
    path = pathlib.Path(filepath)
    if not path.exists():
        print(f"Error: File not found at {filepath}")
        sys.exit(1)

    content = path.read_text(encoding="utf-8").splitlines()
    if line_number > len(content) + 1 or line_number < 1:
        print(f"Error: Line number {line_number} is out of bounds for file with {len(content)} lines.")
        sys.exit(1)

    content.insert(line_number - 1, f"# {comment}")
    path.write_text("\n".join(content), encoding="utf-8")
    print(f"Successfully added comment to {filepath} at line {line_number}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python temp_edit_script.py <filepath> <comment> [line_number]")
        sys.exit(1)

    filepath = sys.argv[1]
    comment = sys.argv[2]
    line_number = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    add_comment_to_file(filepath, comment, line_number)
