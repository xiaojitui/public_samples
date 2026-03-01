import os
import json
import pandas as pd
from collections import defaultdict
from steps.step1_extract_qa import extract_qa
from steps.step2_cluster_questions import cluster_questions
from steps.step3_generate_themes import generate_themes
from steps.step4_cluster_responses import cluster_responses
from prompts.resolution_prompt import RESOLUTION_CHECK_PROMPT
from save_per_call import build_per_call_analysis, save_per_call_analysis

class CallAnalysisAgent:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.transcripts = None
        self.all_qa = []
        self.clustered_questions = None
        self.themes = None
        self.response_buckets = None
        self.clustered_responses = None

    def load_data(self):
        print("Loading data...")
        self.transcripts = pd.read_csv(self.data_path)["transcript"]

    def extract_qa_pairs(self):
        print("Extracting QA pairs...")
        for call in self.transcripts:
            self.all_qa.extend(extract_qa(call))

    def cluster_questions(self):
        print("Clustering questions...")
        questions = [q["question"] for q in self.all_qa]
        self.clustered_questions = cluster_questions(questions)

    def generate_themes(self):
        print("Generating themes...")
        self.themes = generate_themes(self.clustered_questions)

    def aggregate_responses(self):
        print("Aggregating responses...")
        q_map = self.build_question_mapping(self.clustered_questions)
        self.response_buckets = defaultdict(list)
        for qa in self.all_qa:
            raw_q = qa["question"]
            canonical_q = q_map.get(raw_q)
            if canonical_q:
                self.response_buckets[canonical_q].append(qa["agent_response"])

    def cluster_responses(self):
        print("Clustering responses...")
        self.clustered_responses = {
            q: cluster_responses(resps)
            for q, resps in self.response_buckets.items()
        }

    def build_question_mapping(self, clustered_questions):
        mapping = {}
        for group in clustered_questions:
            canonical = group["canonical_question"]
            for variant in group["variants"]:
                mapping[variant] = canonical
        return mapping

    def evaluate_resolutions(self, llm_client):
        print("Evaluating response resolutions...")
        for theme in self.themes:
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
                    r.update({
                        "resolve_question": result["resolve_question"],
                        "resolution_confidence": result["confidence"],
                        "resolution_reasoning": result["reasoning"]
                    })
                    count = r["count"]
                    total_count += count
                    if r["resolve_question"] == 1:
                        resolved_weighted += count
                q["question_resolved_score"] = resolved_weighted / total_count if total_count > 0 else 0.0

    def compute_theme_scores(self):
        print("Computing theme scores...")
        for theme in self.themes:
            total_q_count = 0
            weighted_resolved = 0
            for q in theme["questions"]:
                q_count = q.get("count", 0)
                q_score = q.get("question_resolved_score", 0.0)
                total_q_count += q_count
                weighted_resolved += q_count * q_score
            theme["theme_resolved_score"] = weighted_resolved / total_q_count if total_q_count > 0 else 0.0

    def save_results(self):
        print("Saving results...")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "final_report.json"), "w") as f:
            json.dump(self.themes, f, indent=4)

    def run(self, llm_client):
        self.load_data()
        self.extract_qa_pairs()
        self.cluster_questions()
        self.generate_themes()
        self.aggregate_responses()
        self.cluster_responses()
        self.evaluate_resolutions(llm_client)
        self.compute_theme_scores()
        self.save_results()

if __name__ == "__main__":
    data_path = "data/transcripts.csv"
    output_dir = "outputs/agent_results"
    llm_client = None  # Replace with actual LLM client instance
    agent = CallAnalysisAgent(data_path, output_dir)
    agent.run(llm_client)