import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_list(items, max_tokens, overhead_tokens):
    chunks = []
    current = []
    current_tokens = 0

    for item in items:
        t = count_tokens(item)
        if current_tokens + t + overhead_tokens > max_tokens:
            chunks.append(current)
            current = [item]
            current_tokens = t
        else:
            current.append(item)
            current_tokens += t

    if current:
        chunks.append(current)

    return chunks
