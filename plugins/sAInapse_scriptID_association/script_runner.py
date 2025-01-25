import runpy

def execute_script(script_path):
    try:
        # Execute the __main__ module in the script
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        print(f"Exception during script execution: {e}")
