import sys
# Check if a file path was provided as a command-line argument
def menu(string):
    import json
    return json.dumps([{"label":string,"value":json.dumps({"summary":"No input file detected."})}])
if len(sys.argv) < 2 or sys.argv[1] == "":
    print(menu("Please select a file"), flush=True)
    exit(1)
else:
    file_path = sys.argv[1]
    try:
        with open(file_path, 'r') as f:
            if not f.read():
                print(menu(f"Error: The file {file_path} does not contain text."), flush=True)
                exit(2)
    except FileNotFoundError:
        print(menu(f"Error: The file {file_path} was not found."), flush=True)
        exit(3)
    except PermissionError:
        print(menu(f"Error: Permission denied to access {file_path}."), flush=True)
        exit(4)
    except Exception as e:
        print(menu(f"An error occurred: {e}"), flush=True)
        exit(5)
