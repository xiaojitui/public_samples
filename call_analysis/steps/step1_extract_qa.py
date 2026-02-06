import json
from client import LLMClient

llm = LLMClient()

def extract_qa(conversation_text: str):
    prompt = open("prompts/extract_qa.txt").read().format(
        conversation=conversation_text
    )

    resp = llm.chat([
        {"role": "system", "content": "You extract structured information."},
        {"role": "user", "content": prompt}
    ])

    return json.loads(resp)
