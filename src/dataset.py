from __future__ import annotations
import json
from typing import Dict, List, Optional
from .data_models import ConvFinQARecord

def load_records(path: str) -> List[ConvFinQARecord]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ConvFinQARecord.model_validate(r) for r in raw]

def index_by_id(records: List[ConvFinQARecord]) -> Dict[str, ConvFinQARecord]:
    return {r.id: r for r in records}

def get_turn_question(record: ConvFinQARecord, turn: Optional[int]) -> str:
    # default to the last turn if not specified
    if not record.dialogue.conv_questions:
        return ""
    if turn is None or turn < 0 or turn >= len(record.dialogue.conv_questions):
        turn = len(record.dialogue.conv_questions) - 1
    return record.dialogue.conv_questions[turn]

def get_turn_gold(record: ConvFinQARecord, turn: Optional[int]):
    if not record.dialogue.executed_answers:
        return None
    if turn is None or turn < 0 or turn >= len(record.dialogue.executed_answers):
        turn = len(record.dialogue.executed_answers) - 1
    return record.dialogue.executed_answers[turn]
