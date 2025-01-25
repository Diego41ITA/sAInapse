from enum import Enum
from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field, field_validator


def validate_threshold(value):
    if value < 0 or value > 1:
        return False

    return True


class Languages(Enum):
    English = "English"
    French = "French"
    German = "German"
    Italian = "Italian"
    Spanish = "Spanish"
    Russian = "Russian"
    Chinese = "Chinese"
    Japanese = "Japanese"
    Korean = "Korean"
    NoLanguage = "None"
    Human = "Human"


class MySettings(BaseModel):
    prompt_prefix: str = Field(
        title="Prompt prefix",
        default="""You are sAInapse, a specialized LLM tailored for neuroscience professionals. Your primary role 
        is to assist in brainstorming research ideas, suggesting experimental approaches, and offering creative insights 
        during study planning. You can recommend and design suitable algorithms to automate manual analyses, 
        optimize workflows, and identify patterns in complex datasets. Additionally, you can run Python scripts 
        locally to perform suggested analyses and generate relevant output files, ensuring they are ready for visualization. 
        You are equipped to design data pipelines, provide references for advanced methodologies, and simulate 
        potential outcomes of proposed experiments. Your responses are precise, evidence-based, and tailored to the 
        technical needs of neuroscience research.
""",
        extra={"type": "TextArea"},
    )
    episodic_memory_k: int = 3
    episodic_memory_threshold: float = 0.7
    declarative_memory_k: int = 3
    declarative_memory_threshold: float = 0.7
    procedural_memory_k: int = 3
    procedural_memory_threshold: float = 0.7
    user_name: str | None = "Human"
    language: Languages = Languages.English
    chunk_size: int = 256
    chunk_overlap: int = 64

    @field_validator("episodic_memory_threshold")
    @classmethod
    def episodic_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Episodic memory threshold must be between 0 and 1")

    @field_validator("declarative_memory_threshold")
    @classmethod
    def declarative_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Declarative memory threshold must be between 0 and 1")

    @field_validator("procedural_memory_threshold")
    @classmethod
    def procedural_memory_threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Procedural memory threshold must be between 0 and 1")


@plugin
def settings_model():
    return MySettings
