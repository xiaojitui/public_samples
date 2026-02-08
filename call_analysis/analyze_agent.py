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



def summarize_agent(agent_id, agent_data):
    llm = LLMClient()

    prompt = f"""
You are analyzing call-handling behavior for an insurance call center agent.

Agent ID: {agent_id}

Data:
{json.dumps(agent_data, indent=2)}

Provide:
1) Main themes this agent handled
2) Common questions they received
3) Notable response patterns or differences
4) One coaching suggestion (if any)

Be concise and factual.
"""

    return llm.chat([
        {"role": "system", "content": "You analyze agent performance."},
        {"role": "user", "content": prompt}
    ])


def generate_agent_reports(
    all_qa,
    clustered_questions,
    themes,
    clustered_responses,
    generate_llm_summary=False
):
    agent_stats = build_agent_stats(
        all_qa,
        clustered_questions,
        themes,
        clustered_responses
    )

    reports = {}

    for agent_id, data in agent_stats.items():
        report = {
            "agent_id": agent_id,
            "themes": []
        }

        for theme, questions in data.items():
            theme_block = {
                "theme": theme,
                "questions": []
            }

            for q, responses in questions.items():
                theme_block["questions"].append({
                    "question": q,
                    "responses": [
                        {"response": r, "count": c}
                        for r, c in responses.items()
                    ]
                })

            report["themes"].append(theme_block)

        if generate_llm_summary:
            report["summary"] = summarize_agent(agent_id, report["themes"])

        reports[agent_id] = report

    return reports


'''
{
  "agent_id": "ABC1234",
  "themes": [
    {
      "theme": "Billing & Payments",
      "questions": [
        {
          "question": "How do I pay my insurance bill?",
          "responses": [
            {
              "response": "You can pay online through the customer portal.",
              "count": 12
            },
            {
              "response": "Payments can be made by phone or online.",
              "count": 3
            }
          ]
        }
      ]
    }
  ],
  "summary": "This agent primarily handled billing inquiries and consistently directed customers to the online portal..."
}
'''

from prompts.resolution_prompt import RESOLUTION_CHECK_PROMPT

def evaluate_agent_resolution(llm_client, agent_report):

    for theme in agent_report["themes"]:

        for q in theme["questions"]:

            total_count = 0
            resolved_weighted = 0

            for r in q["responses"]:

                prompt = RESOLUTION_CHECK_PROMPT.format(
                    question=q["question"],
                    response=r["response"]
                )

                result = llm_client.chat_json(prompt)

                r["resolve_question"] = result["resolve_question"]
                r["resolution_confidence"] = result["confidence"]
                r["resolution_reasoning"] = result["reasoning"]

                count = r["count"]
                total_count += count

                if r["resolve_question"] == 1:
                    resolved_weighted += count

            if total_count > 0:
                q["question_resolved_score"] = resolved_weighted / total_count
            else:
                q["question_resolved_score"] = 0.0

    return agent_report

# agent_report = evaluate_agent_resolution(llm_client, agent_report)

