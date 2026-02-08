import pandas as pd
from steps.step1_extract_qa import extract_qa
from steps.step2_cluster_questions import cluster_questions
from steps.step3_generate_themes import generate_themes
from steps.step4_cluster_responses import cluster_responses

from prompts.resolution_prompt import RESOLUTION_CHECK_PROMPT

df = pd.read_csv("data/transcripts.csv")

all_qa = []
for call in df["transcript"]:
    all_qa.extend(extract_qa(call))

questions = [q["question"] for q in all_qa]
clustered_qs = cluster_questions(questions)

themes = generate_themes(clustered_qs)

# # responses per question
# response_map = {}
# for qa in all_qa:
#     response_map.setdefault(qa["question"], []).append(qa["agent_response"])
# clustered_responses = {
#     q: cluster_responses(resps)
#     for q, resps in response_map.items()
# }


def build_question_mapping(clustered_questions):
    mapping = {}
    for group in clustered_questions:
        canonical = group["canonical_question"]
        for variant in group["variants"]:
            mapping[variant] = canonical
    return mapping

from collections import defaultdict

def aggregate_responses_by_canonical_question(all_qa, clustered_questions):
    q_map = build_question_mapping(clustered_questions)
    response_buckets = defaultdict(list)

    for qa in all_qa:
        raw_q = qa["question"]
        canonical_q = q_map.get(raw_q)
        if canonical_q:
            response_buckets[canonical_q].append(qa["agent_response"])

    return response_buckets

response_buckets = aggregate_responses_by_canonical_question(
    all_qa,
    clustered_questions
)


clustered_responses = {
    q: cluster_responses(resps)
    for q, resps in response_buckets.items()
}


'''
{
  "theme": "Billing & Payments",
  "questions": [
    {
      "question": "How do I pay my insurance bill?",
      "count": 42,
      "responses": [
        {
          "response": "You can pay online through the customer portal.",
          "count": 31,
          "confidence": 0.96
        }
      ]
    }
  ]
}
'''

def evaluate_response_resolution(llm_client, final_report):

    for theme in final_report:

        for q in theme["questions"]:

            question_text = q["question"]

            total_count = 0
            resolved_weighted = 0

            for r in q["responses"]:

                prompt = RESOLUTION_CHECK_PROMPT.format(
                    question=question_text,
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

    return final_report

def compute_theme_resolved_scores(final_report):

    for theme in final_report:

        total_q_count = 0
        weighted_resolved = 0

        for q in theme["questions"]:
            q_count = q.get("count", 0)
            q_score = q.get("question_resolved_score", 0.0)

            total_q_count += q_count
            weighted_resolved += q_count * q_score

        if total_q_count > 0:
            theme["theme_resolved_score"] = weighted_resolved / total_q_count
        else:
            theme["theme_resolved_score"] = 0.0

    return final_report


'''
final_report = build_final_report(...)
final_report = evaluate_response_resolution(llm_client, final_report)
final_report = compute_theme_resolved_scores(final_report)
'''

'''
{
  "theme": "Billing & Payments",
  "theme_resolved_score": 0.83,
  "questions": [
    {
      "question": "How do I pay my insurance bill?",
      "count": 42,
      "question_resolved_score": 0.86,
      "responses": [
        {
          "response": "You can pay online through the customer portal.",
          "count": 31,
          "confidence": 0.96,
          "resolve_question": 1,
          "resolution_confidence": 0.94,
          "resolution_reasoning": "Directly provides payment method"
        }
      ]
    }
  ]
}
'''

from save_per_call import build_per_call_analysis, save_per_call_analysis

per_call_data = build_per_call_analysis(transcripts, clustered_qs)

save_per_call_analysis(per_call_data)

import json

calls = json.load(open("outputs/per_call_analysis.json"))

for c in calls:
    scores = [q["question_resolved_score"] for q in c["questions"]]
    c["call_resolution_score"] = sum(scores)/len(scores) if scores else 1

sorted(calls, key=lambda x: x["call_resolution_score"])[0:20]

import pandas as pd

pd.json_normalize(per_call_data, record_path="questions",
                  meta=["call_id","agent_id"]).to_parquet("outputs/per_call_analysis.parquet")




