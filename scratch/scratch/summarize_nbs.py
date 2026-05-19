import json, sys

for path in sys.argv[1:]:
    print(f"\n--- Notebook: {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            for i, cell in enumerate(nb.get('cells', [])):
                ctype = cell.get('cell_type')
                source = "".join(cell.get('source', []))
                preview = source[:150].replace('\n', ' ') + ('...' if len(source) > 150 else '')
                print(f"Cell {i} [{ctype}]: {preview}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
