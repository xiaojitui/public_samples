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


call_analysis/
├── README.md
├── requirements.txt
├── client.py                # already exists (you said)
├── config.py
├── run_pipeline.py
├── data/
│   └── transcripts.csv
├── prompts/
│   ├── extract_qa.txt
│   ├── cluster_questions.txt
│   ├── generate_themes.txt
│   └── cluster_responses.txt
├── steps/
│   ├── step1_extract_qa.py
│   ├── step2_cluster_questions.py
│   ├── step3_generate_themes.py
│   └── step4_cluster_responses.py
└── utils/
    ├── io.py
    └── chunking.py

