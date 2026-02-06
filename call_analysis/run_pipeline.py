import pandas as pd
from steps.step1_extract_qa import extract_qa
from steps.step2_cluster_questions import cluster_questions
from steps.step3_generate_themes import generate_themes
from steps.step4_cluster_responses import cluster_responses

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

clustered_responses = {
    canonical_q: cluster_responses(responses)
    for canonical_q, responses in response_buckets.items()
}
