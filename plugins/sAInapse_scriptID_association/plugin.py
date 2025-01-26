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

    # flag = cat.llm(f"If the input is a request for scriptID association, you MUST answer ONLY yes in lower case; otherwise, answer ONLY no. Input --> {cat.working_memory.user_message_json.text}")

    # if flag == "yes":
    #     prefix = """" 
    #     You are ONLY specialized in ONE task. You need to answer ONLY with the taskID that you find in the documentation provided. I don't want complex
    #     answers, just the taskID. If you don't find any taskID, answer 'no'.
    #     """
    # else:
    #     prefix = """" You are sAInapse, a specialized LLM tailored for neuroscience professionals. Your primary role 
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

# @hook
# def before_cat_sends_message(message, cat):
#     flag = cat.llm(f"If the input is a request for scriptID association, you MUST answer ONLY yes in lower case; otherwise, answer ONLY no. Input --> {cat.working_memory.user_message_json.text}")
    
#     if flag == "yes":
#         scriptID = message['content']
#         if scriptID == "no":
#             error_message = "I'm sorry, i didn't found any script that can satify your request. Please try again."
#             return error_message

#         # # Define the base directory where your main script is located
#         # this_dir = os.path.abspath(os.path.dirname(__file__))

#         # scripts = os.listdir(this_dir)  # List all files in the directory

#         # # Add the .py extension if not already included
#         # script_name = f"{scriptID}.py" if not scriptID.endswith(".py") else scriptID

#         # if script_name in scripts:
#         #     execute_script(os.path.join(this_dir, script_name))
#         #     return f"Match founded. Script with ID {scriptID} executed successfully."
#         # else:
#         #     return f"Script with ID {scriptID} not found."

#         if scriptID == "script1":
#             prova()
#         elif scriptID == "script2":
#             prova()

#         return f"Match founded. Script with ID {scriptID} executed successfully."
#     else:
#         return message

def script1():
# URL of the zip file
    url = "https://ninapro.hevs.ch/files/DB5_Preproc/s1.zip"  # Example for s1

    # Specify the folder where you want to download the zip file
    download_folder = "/app/cat/data/datasets"  # Change this to your desired folder
    os.makedirs(download_folder, exist_ok=True)  # Make sure the folder exists


    zip_filename = "s1.zip"

    # Download the file if it doesn't already exist
    if not os.path.exists(zip_filename):
        print(f"Downloading {url} to {zip_filename} ...")
        r = requests.get(url, allow_redirects=True)
        with open(zip_filename, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print(f"{zip_filename} already present, skipping download.")

    # Specify the folder where you want to extract the contents
    extract_folder = "/app/cat/data/datasets"  
    os.makedirs(extract_folder, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_filename, 'r') as zf:
        zf.extractall(extract_folder)

    print(f"Files extracted to {extract_folder}")

def script2():
    mat_files = []
    for root, dirs, files in os.walk("/app/cat/data/datasets/s1"):
        for f in files:
            if f.endswith(".mat"):
                mat_files.append(os.path.join(root, f))

    if not mat_files:
        raise FileNotFoundError("No .mat files found. Check the unzipped structure.")

    demo_mat = mat_files[0]
    print(f"\nUsing: {demo_mat}")

    data = loadmat(demo_mat)
    print("Keys in .mat:", list(data.keys()))

    emg_signal = data['emg']
    labels = data['restimulus'].flatten()

    print("EMG shape:", emg_signal.shape)
    print("Unique labels:", np.unique(labels))

    plt.figure(figsize=(12,4))
    plt.plot(emg_signal[:1000, 0], label="Channel 0")
    plt.title("First 1000 Samples of EMG Channel 0")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    # Prova a scrivere direttamente su BytesIO
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)

    if img_bytes.getvalue():
        print("L'immagine è stata scritta correttamente in memoria.")
    else:
        print("Errore nella scrittura dell'immagine in memoria.")
        return

    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    html_response = f'''
    <html>
        <body>
            <h1>Grafico generato con successo</h1>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
    </html>
    '''

    # Salva il file HTML fuori dal container, se vuoi
    output_path = "/app/cat/data/plots/grafico.html"
    
    try:
        with open(output_path, "w") as f:
            f.write(html_response)
            print(f"File {output_path} scritto con successo.")
    except Exception as e:
        print(f"Errore nel salvataggio del file: {e}")

def prova():
    with open('/app/cat/data/outcome_data/script2.txt', 'w') as file:
        file.write('ciao')

@tool(return_direct=True)
def test_tool(tool_input, cat):
    """
        This tool performs scriptID association for the user requested task.
        The tool input is the task requested by the user.
    """

    # Load the PDF and extract content
    pdf_path = "/app/cat/data/scripts_docs/scripts_doc.pdf"
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


    # Get the path to the outcome_data folder
    outcome_data_directory = "/app/cat/data/outcome_data"

    # Define the path for the new file
    file_path = os.path.join(outcome_data_directory, "test.txt")

    create_file(file_path, tool_input)
    

    return f"File 'test.txt' has been created in {outcome_data_directory} with the provided content."
