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

# responses per question
response_map = {}
for qa in all_qa:
    response_map.setdefault(qa["question"], []).append(qa["agent_response"])

clustered_responses = {
    q: cluster_responses(resps)
    for q, resps in response_map.items()
}
