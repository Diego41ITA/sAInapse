from cat.mad_hatter.decorators import hook, tool
import os
from .script_runner import execute_script
import zipfile
import requests
import os
import numpy as np
from scipy.io import loadmat
import io
import base64
import matplotlib.pyplot as plt

from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@hook
def agent_prompt_prefix(prefix, cat):

    prefix = """" You are sAInapse, a specialized LLM tailored for neuroscience professionals. Your primary role 
    #     is to assist in brainstorming research ideas, suggesting experimental approaches, and offering creative insights 
    #     during study planning. You can recommend and design suitable algorithms to automate manual analyses, 
    #     optimize workflows, and identify patterns in complex datasets. Additionally, you can run Python scripts 
    #     locally to perform suggested analyses and generate relevant output files, ensuring they are ready for visualization. 
    #     You are equipped to design data pipelines, provide references for advanced methodologies, and simulate 
    #     potential outcomes of proposed experiments. Your responses are precise, evidence-based, and tailored to the 
    #     technical needs of neuroscience research. You can search for correspondences between the user request for a specific 
    #     task to be executed and a set of documentation provided as input. If a match is found, you inform the user, citing the 
    #     relevant script ID associated with the required action. This script ID corresponds to deterministic code that can be 
    #     executed by internal logic provided by LangChain, enabling local computation to perform the suggested analysis. 
    # """

    return prefix


@tool(return_direct=True)
def test_tool(tool_input, cat):
    """
        This tool activates ONLY and ALWAYS when a message start with @. The tool input is ALWAYS the request specified after the @.
        This tool will react to specific task requests, not to general questions.
    """

    # Load the PDF and extract content
    pdf_path = "/scripts_docs/scripts_doc.pdf"
    content = extract_text_from_pdf(pdf_path)

    # association prompt
    prompt = f"""You are ONLY specialized in ONE task. You need to perform a scriptID association beetween the user request
                 and the documentation provided later. You need to answer ONLY with the taskID that you find in the documentation provided. 
                 I don't want complex answers, just the taskID. If you don't find any taskID, answer 'no' in lower case. You will find the scriptID
                 to pick before every corresponding explanation of the task.
                User request -> {tool_input}.
                documentation -> {content}.
                """
    
    #run the LLM
    scriptID = cat.llm(prompt, stream=True)

    
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
