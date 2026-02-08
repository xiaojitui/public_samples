import json
from collections import defaultdict


def compute_question_resolved_score(responses):

    if not responses:
        return 0

    resolved = [r["resolve_question"] for r in responses]
    return sum(resolved) / len(resolved)


def build_per_call_analysis(transcripts, clustered_qs):

    """
    transcripts:
        list of transcript objects already loaded

    clustered_qs:
        output after clustering and mapping responses
    """

    calls = defaultdict(lambda: {
        "call_id": None,
        "agent_id": None,
        "questions": []
    })

    for item in clustered_qs:

        call_id = item["call_id"]
        agent_id = item["agent_id"]

        calls[call_id]["call_id"] = call_id
        calls[call_id]["agent_id"] = agent_id

        q_entry = {
            "question": item["clustered_question"],
            "theme": item["theme"],
            "responses": item["responses"],
        }

        q_entry["question_resolved_score"] = compute_question_resolved_score(
            item["responses"]
        )

        calls[call_id]["questions"].append(q_entry)

    return list(calls.values())


def save_per_call_analysis(per_call_data, path="outputs/per_call_analysis.json"):

    with open(path, "w") as f:
        json.dump(per_call_data, f, indent=2)
