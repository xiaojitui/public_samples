import json
from client import LLMClient

llm = LLMClient()

def cluster_questions(questions: list[str]):
    prompt = open("prompts/cluster_questions.txt").read().format(
        questions="\n".join(questions)
    )

    resp = llm.chat([
        {"role": "system", "content": "You cluster questions."},
        {"role": "user", "content": prompt}
    ])

    return json.loads(resp)
