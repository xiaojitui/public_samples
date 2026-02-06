# Stage 1 — Extract structured Q/A pairs per call

LLM is best at:

Identifying user questions

Identifying agent responses

Normalizing phrasing

🎯 Output: canonicalized question + canonicalized response


# Stage 2 — Aggregate common user questions

We cluster questions across calls using the LLM (semantic dedup).

🎯 Output:

canonical_question

count


# Stage 3 — Generate short list of themes

We ask the LLM to produce 5–10 themes max, then map questions → themes.

🎯 Output:

theme

questions under theme

counts

# Stage 4 — Aggregate agent responses per question

Cluster agent responses per question and count variants.

🎯 Output:

question

response_variant

count
