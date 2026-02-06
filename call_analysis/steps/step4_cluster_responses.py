import json
from client import LLMClient

llm = LLMClient()

def cluster_responses(responses: list[str]):
    prompt = open("prompts/cluster_responses.txt").read().format(
        responses="\n".join(responses)
    )

    resp = llm.chat([
        {"role": "system", "content": "You cluster agent responses."},
        {"role": "user", "content": prompt}
    ])

    return json.loads(resp)
