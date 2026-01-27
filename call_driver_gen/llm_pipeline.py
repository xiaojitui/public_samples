import json
from openai import AzureOpenAI
from config import *
from utils_token_chunking import count_tokens

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

MODEL = AZURE_OPENAI_DEPLOYMENT


# ---------------- MAP STEP ----------------
def extract_questions_from_chunk(chunk, chunk_id):
    prompt = f"""
You are analyzing insurance acquisition documents.

Based ONLY on the text below, generate:

- High-level customer question categories
- Under each category, realistic customer questions

Do NOT generate answers yet.
Do NOT use outside knowledge.
Focus only on customer concerns implied by the text.

TEXT:
\"\"\"{chunk}\"\"\"

OUTPUT FORMAT (JSON):
[
  {{
    "category": "Category name",
    "questions": ["question 1", "question 2"]
  }}
]
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You extract customer concerns from insurance communications."},
            {"role": "user", "content": prompt}
        ]
    )

    print(f"✓ Chunk {chunk_id} processed | Tokens: {count_tokens(chunk)}")
    return json.loads(response.choices[0].message.content)


# ---------------- REDUCE STEP ----------------
def consolidate_categories(all_chunk_outputs):
    combined_text = json.dumps(all_chunk_outputs, indent=2)

    prompt = f"""
You are consolidating customer question categories.

Merge similar categories, remove duplicate questions,
and keep only the TOP 10 most important categories
with up to 20 strong representative questions each.

INPUT:
{combined_text}

OUTPUT FORMAT (JSON):
[
  {{
    "category": "Final category name",
    "questions": ["question 1", "question 2", ...]
  }}
]
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You consolidate overlapping taxonomies."},
            {"role": "user", "content": prompt}
        ]
    )

    print("✓ Categories consolidated")
    return json.loads(response.choices[0].message.content)


# ---------------- FINAL ANSWERS ----------------
def generate_answers(consolidated_categories):
    prompt = f"""
You are an insurance call center training expert.

For EACH question below, provide a professional, empathetic agent response.
If information is uncertain, guide the customer to appropriate support instead of guessing.

INPUT:
{json.dumps(consolidated_categories, indent=2)}

OUTPUT FORMAT (JSON):
[
  {{
    "category": "...",
    "questions": [
      {{
        "question": "...",
        "agent_response": "..."
      }}
    ]
  }}
]
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You write compliant insurance agent responses."},
            {"role": "user", "content": prompt}
        ]
    )

    print("✓ Agent answers generated")
    return json.loads(response.choices[0].message.content)
