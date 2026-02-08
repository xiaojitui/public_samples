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


Theme
 └── Canonical Question (clustered)
       └── Agent Response Variants (clustered)


    For each agent:

What themes did they receive calls about?

What questions did they handle under each theme?

How did they respond, and how often?

Where does their response pattern differ from peers?




Layer 1: Semantic schema (slow-changing)
  - Themes
  - Canonical questions
  - Canonical responses

Layer 2: Facts (append-only)
  - (transcript_id, agent_id, canonical_question, canonical_response)

Layer 3: Aggregates (recomputable)
  - final_report
  - agent_report



What should happen when NEW transcripts arrive?
Step 1 — Run ONLY the cheap extraction on new data

For new transcripts only:

Extract Q/A pairs

Keep agent_id, transcript_id

Add confidence + reasoning (as before)

No clustering yet.

Step 2 — Map new questions to EXISTING canonical questions

For each new question:

Semantic match against existing canonical questions

If similarity ≥ threshold → assign

Else → mark as unmapped

This is a retrieval problem, not a clustering problem.

✅ Fast
✅ Stable
✅ Deterministic

Step 3 — Map agent responses to canonical responses

Same logic:

Match against existing canonical responses per question

Add new variant only if confidence is low

Step 4 — Update aggregates (no LLM needed)

Now update:

final_report counts

agent_report counts

Pure Python.

When do you EVER rerun clustering?

Only when semantic drift exceeds tolerance.

Examples:

10–15% of new questions are unmapped

A theme grows too large or incoherent

Business introduces new policy/products

This is a controlled re-index, not a daily job.




We will:

Add LLM-based resolution classification

Add field resolve_question

Compute:
    question_resolved_score =
  sum(resolve_question * response_count) / total_response_count

(Weighted by counts — this is important and correct statistically.)


##
low_perf_questions = {
    q["question"]
    for theme in final_report
    for q in theme["questions"]
    if q["question_resolved_score"] < 0.5
}
bad_calls = []

for call in call_reports:
    for q in call["questions"]:
        if q["question"] in low_perf_questions:
            bad_calls.append(call)
            break
##
low_agents = [
    r["agent_id"]
    for r in agent_reports
    if r["overall_resolution_score"] < 0.6
]
##

