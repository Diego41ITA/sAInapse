from cat.mad_hatter.decorators import hook
import os
from .script_runner import execute_script

@hook
def agent_prompt_prefix(prefix, cat):

    flag = cat.llm(f"If the input is a request for scriptID association, you MUST answer ONLY yes in lower case; otherwise, answer ONLY no. Input --> {cat.working_memory.user_message_json.text}")

    if flag == "yes":
        prefix = """" 
        You are ONLY specialized in ONE task. You need to answer ONLY with the taskID that you find in the documentation provided. I don't want complex
        answers, just the taskID. If you don't find any taskID, answer 'no'.
        """
    else:
        prefix = """" You are sAInapse, a specialized LLM tailored for neuroscience professionals. Your primary role 
        is to assist in brainstorming research ideas, suggesting experimental approaches, and offering creative insights 
        during study planning. You can recommend and design suitable algorithms to automate manual analyses, 
        optimize workflows, and identify patterns in complex datasets. Additionally, you can run Python scripts 
        locally to perform suggested analyses and generate relevant output files, ensuring they are ready for visualization. 
        You are equipped to design data pipelines, provide references for advanced methodologies, and simulate 
        potential outcomes of proposed experiments. Your responses are precise, evidence-based, and tailored to the 
        technical needs of neuroscience research. You can search for correspondences between the user request for a specific 
        task to be executed and a set of documentation provided as input. If a match is found, you inform the user, citing the 
        relevant script ID associated with the required action. This script ID corresponds to deterministic code that can be 
        executed by internal logic provided by LangChain, enabling local computation to perform the suggested analysis. 
    """

    return prefix

@hook
def before_cat_sends_message(message, cat):
    flag = cat.llm(f"If the input is a request for scriptID association, you MUST answer ONLY yes in lower case; otherwise, answer ONLY no. Input --> {cat.working_memory.user_message_json.text}")
    
    if flag == "yes":
        scriptID = message['content']
        if scriptID == "no":
            error_message = "I'm sorry, i didn't found any script that can satify your request. Please try again."
            return error_message

        # Define the base directory where your main script is located
        this_dir = os.path.abspath(os.path.dirname(__file__))

        scripts = os.listdir(this_dir)  # List all files in the directory

        # Add the .py extension if not already included
        script_name = f"{scriptID}.py" if not scriptID.endswith(".py") else scriptID

        if script_name in scripts:
            execute_script(os.path.join(this_dir, script_name))
            return f"Match founded. Script with ID {scriptID} executed successfully."
        else:
            return f"Script with ID {scriptID} not found."
    else:
        return message

# @tool
# def test_tool(tool_input, cat):
#     """
#         This tool performs scriptID association for the user requested task.
#         The tool input is the task requested by the user.
#     """

#     # association prompt
#     prompt = f"""Find the script ID associated with the task:
#                 {tool_input}.
#                 by inspecting the documentation provided here:
#                 {cat.working_memory.declarative_memories}.

#                 You MUST answer only with the script ID in the same exact way as the documentation, e.g. "script" followed by a number. If you don't know the script ID, type 'no'.
#                 """
    
#     #run the LLM
#     scriptID = cat.llm(prompt, stream=True)

    
#     if scriptID == "no":
#         error_message = "I'm sorry, i didn't found any script that can satify your request. Please try again."
#         return error_message

#     # Define the base directory where your main script is located
#     this_dir = os.path.abspath(os.path.dirname(__file__))

#     scripts = os.listdir(this_dir)  # List all files in the directory
    
#     # Add the .py extension if not already included
#     script_name = f"{scriptID}.py" if not scriptID.endswith(".py") else scriptID

#     if script_name in scripts:
#         execute_script(os.path.join(this_dir, script_name))
#         return f"Match founded. Script with ID {scriptID} executed successfully."
#     else:
#         return f"Script with ID {scriptID} not found."


    # # Get the path to the outcome_data folder
    # outcome_data_directory = "/app/cat/data/outcome_data"

    # # Define the path for the new file
    # file_path = os.path.join(outcome_data_directory, "test.txt")

    # create_file(file_path, tool_input)
    

    # return f"File 'test.txt' has been created in {outcome_data_directory} with the provided content."
