from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.tools import Tool
import pandas as pd
import os
import json
from steps.step1_extract_qa import extract_qa
from steps.step2_cluster_questions import cluster_questions
from steps.step3_generate_themes import generate_themes
from steps.step4_cluster_responses import cluster_responses
from save_per_call import build_per_call_analysis, save_per_call_analysis
from langchain.graphs import Graph

# Define tools

def load_data_tool(data_path):
    """Load transcripts from a CSV file."""
    print("Loading data...")
    return pd.read_csv(data_path)["transcript"]

def extract_qa_tool(transcripts):
    """Extract QA pairs from transcripts."""
    print("Extracting QA pairs...")
    all_qa = []
    for call in transcripts:
        all_qa.extend(extract_qa(call))
    return all_qa

def cluster_questions_tool(all_qa):
    """Cluster questions."""
    print("Clustering questions...")
    questions = [q["question"] for q in all_qa]
    return cluster_questions(questions)

def generate_themes_tool(clustered_questions):
    """Generate themes from clustered questions."""
    print("Generating themes...")
    return generate_themes(clustered_questions)

def aggregate_responses_tool(all_qa, clustered_questions):
    """Aggregate responses by canonical question."""
    print("Aggregating responses...")
    from collections import defaultdict

    def build_question_mapping(clustered_questions):
        mapping = {}
        for group in clustered_questions:
            canonical = group["canonical_question"]
            for variant in group["variants"]:
                mapping[variant] = canonical
        return mapping

    q_map = build_question_mapping(clustered_questions)
    response_buckets = defaultdict(list)
    for qa in all_qa:
        raw_q = qa["question"]
        canonical_q = q_map.get(raw_q)
        if canonical_q:
            response_buckets[canonical_q].append(qa["agent_response"])
    return response_buckets

def cluster_responses_tool(response_buckets):
    """Cluster responses for each canonical question."""
    print("Clustering responses...")
    return {
        q: cluster_responses(resps)
        for q, resps in response_buckets.items()
    }

def save_results_tool(themes, output_dir):
    """Save the final themes to a JSON file."""
    print("Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "final_report.json"), "w") as f:
        json.dump(themes, f, indent=4)

# Initialize tools

data_tool = Tool(
    name="Load Data",
    func=load_data_tool,
    description="Load transcripts from a CSV file."
)

qa_tool = Tool(
    name="Extract QA Pairs",
    func=extract_qa_tool,
    description="Extract QA pairs from transcripts."
)

question_cluster_tool = Tool(
    name="Cluster Questions",
    func=cluster_questions_tool,
    description="Cluster questions from QA pairs."
)

theme_tool = Tool(
    name="Generate Themes",
    func=generate_themes_tool,
    description="Generate themes from clustered questions."
)

response_aggregate_tool = Tool(
    name="Aggregate Responses",
    func=aggregate_responses_tool,
    description="Aggregate responses by canonical question."
)

response_cluster_tool = Tool(
    name="Cluster Responses",
    func=cluster_responses_tool,
    description="Cluster responses for each canonical question."
)

save_tool = Tool(
    name="Save Results",
    func=save_results_tool,
    description="Save the final themes to a JSON file."
)

# Define a graph-based workflow

def create_graph_workflow():
    graph = Graph()

    # Define nodes
    graph.add_node("load_data", load_data_tool, description="Load transcripts from a CSV file.")
    graph.add_node("extract_qa", extract_qa_tool, description="Extract QA pairs from transcripts.")
    graph.add_node("cluster_questions", cluster_questions_tool, description="Cluster questions from QA pairs.")
    graph.add_node("generate_themes", generate_themes_tool, description="Generate themes from clustered questions.")
    graph.add_node("aggregate_responses", aggregate_responses_tool, description="Aggregate responses by canonical question.")
    graph.add_node("cluster_responses", cluster_responses_tool, description="Cluster responses for each canonical question.")
    graph.add_node("save_results", save_results_tool, description="Save the final themes to a JSON file.")

    # Define edges
    graph.add_edge("load_data", "extract_qa")
    graph.add_edge("extract_qa", "cluster_questions")
    graph.add_edge("cluster_questions", "generate_themes")
    graph.add_edge("generate_themes", "aggregate_responses")
    graph.add_edge("aggregate_responses", "cluster_responses")
    graph.add_edge("cluster_responses", "save_results")

    return graph

# Initialize the graph workflow
workflow_graph = create_graph_workflow()

if __name__ == "__main__":
    data_path = "data/transcripts.csv"
    output_dir = "outputs/agent_results_langchain"

    # Run the graph workflow
    workflow_graph.run({
        "load_data": {"data_path": data_path},
        "save_results": {"output_dir": output_dir}
    })