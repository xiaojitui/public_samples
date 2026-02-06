import json
from client import LLMClient

llm = LLMClient()

def generate_themes(clustered_questions):
    prompt = open("prompts/generate_themes.txt").read().format(
        questions_json=json.dumps(clustered_questions, indent=2)
    )

    resp = llm.chat([
        {"role": "system", "content": "You generate concise themes."},
        {"role": "user", "content": prompt}
    ])

    return json.loads(resp)
