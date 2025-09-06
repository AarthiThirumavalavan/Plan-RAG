from __future__ import annotations
from typing import Dict, List, Union
from pydantic import BaseModel, Field

# Mirrors the dataset card's definitions
# (ConvFinQARecord, Document, Dialogue, Features)
# so parsing is robust to the cleaned structure.  :contentReference[oaicite:1]{index=1}

Number = Union[float, int, str]

class Document(BaseModel):
    pre_text: str = Field(default="")
    post_text: str = Field(default="")
    table: Dict[str, Dict[str, Number]] = Field(default_factory=dict)

class Dialogue(BaseModel):
    conv_questions: List[str] = Field(default_factory=list)
    conv_answers: List[str] = Field(default_factory=list)
    turn_program: List[str] = Field(default_factory=list)
    executed_answers: List[Number] = Field(default_factory=list)
    qa_split: List[bool] = Field(default_factory=list)

class Features(BaseModel):
    num_dialogue_turns: int = 0
    has_type2_question: bool = False
    has_duplicate_columns: bool = False
    has_non_numeric_values: bool = False

class ConvFinQARecord(BaseModel):
    id: str
    doc: Document
    dialogue: Dialogue
    features: Features
