import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
from config import MODEL_NAME, MAX_MODEL_TOKENS, SAFETY_MARGIN

nltk.download("punkt")

encoding = tiktoken.encoding_for_model(MODEL_NAME)

SYSTEM_PROMPT_BASE = "You are an insurance domain analyst extracting customer concerns."
USER_PROMPT_TEMPLATE = """
Analyze the following insurance acquisition document chunk and extract likely customer question categories and questions.

TEXT:
\"\"\"{chunk}\"\"\"
"""

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


SYSTEM_TOKENS = count_tokens(SYSTEM_PROMPT_BASE)
USER_TEMPLATE_TOKENS = count_tokens(USER_PROMPT_TEMPLATE.format(chunk=""))


def chunk_by_tokens(text: str):
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        projected_tokens = (
            current_tokens
            + sent_tokens
            + SYSTEM_TOKENS
            + USER_TEMPLATE_TOKENS
            + SAFETY_MARGIN
        )

        if projected_tokens > MAX_MODEL_TOKENS:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent
            current_tokens = sent_tokens
        else:
            current_chunk += " " + sent
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
