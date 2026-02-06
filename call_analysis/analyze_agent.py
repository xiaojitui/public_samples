import json
from collections import defaultdict
from typing import Dict, List
from client import LLMClient


def build_question_to_canonical(clustered_questions):
    mapping = {}
    for group in clustered_questions:
        canonical = group["canonical_question"]
        for v in group["variants"]:
            mapping[v] = canonical
    return mapping


def build_question_to_theme(themes):
    mapping = {}
    for theme_block in themes:
        theme = theme_block["theme"]
        for q in theme_block["questions"]:
            mapping[q["question"]] = theme
    return mapping


def build_agent_stats(
    all_qa: List[dict],
    clustered_questions: List[dict],
    themes: List[dict],
    clustered_responses: Dict[str, List[dict]]
):
    q_to_canonical = build_question_to_canonical(clustered_questions)
    q_to_theme = build_question_to_theme(themes)

    agent_stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    for qa in all_qa:
        agent = qa["agent_id"]
        raw_q = qa["question"]
        response = qa["agent_response"]

        canonical_q = q_to_canonical.get(raw_q)
        theme = q_to_theme.get(canonical_q)

        if not canonical_q or not theme:
            continue

        agent_stats[agent][theme][canonical_q][response] += 1

    return agent_stats
