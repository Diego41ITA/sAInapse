from cat.mad_hatter.decorators import tool, hook


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

@hook
def before_cat_recalls_episodic_memories(default_episodic_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_episodic_recall_config["k"] = settings["episodic_memory_k"]
    default_episodic_recall_config["threshold"] = settings["episodic_memory_threshold"]

    return default_episodic_recall_config


@hook
def before_cat_recalls_declarative_memories(default_declarative_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_declarative_recall_config["k"] = settings["declarative_memory_k"]
    default_declarative_recall_config["threshold"] = settings[
        "declarative_memory_threshold"
    ]

    return default_declarative_recall_config


@hook
def before_cat_recalls_procedural_memories(default_procedural_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_procedural_recall_config["k"] = settings["procedural_memory_k"]
    default_procedural_recall_config["threshold"] = settings[
        "procedural_memory_threshold"
    ]

    return default_procedural_recall_config


@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    username = settings["user_name"] if settings["user_name"] != "" else "Human"
    suffix = f"""
# Context

{{episodic_memory}}

{{declarative_memory}}

{{tools_output}}
"""

    if settings["language"] == "Human":
        suffix += f"""
ALWAYS answer in the {username}'s language
"""
    elif settings["language"] not in ["None", "Human"]:
        suffix += f"""
ALWAYS answer in {settings["language"]}
"""

    suffix += f"""
## Conversation until now:"""

    return suffix

@hook
def rabbithole_instantiates_splitter(text_splitter, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    text_splitter._chunk_size = settings["chunk_size"]
    text_splitter._chunk_overlap = settings["chunk_overlap"]
    return text_splitter


