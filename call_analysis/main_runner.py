# this should be at bottom of run_pipeline.py


def main():

    # 1. Load transcripts
    df = pd.read_csv("data/transcripts.csv")

    # 2. Extract QA pairs from each call
    all_qa = []
    for call in df["transcript"]:
        all_qa.extend(extract_qa(call))

    # 3. Cluster questions
    questions = [q["question"] for q in all_qa]
    clustered_questions = cluster_questions(questions)

    # 4. Generate themes
    themes = generate_themes(clustered_questions)

    # 5. Aggregate and cluster responses by canonical question
    response_buckets = aggregate_responses_by_canonical_question(
        all_qa,
        clustered_questions
    )

    clustered_responses = {
        q: cluster_responses(resps)
        for q, resps in response_buckets.items()
    }

    # 6. Build final report structure
    final_report = build_final_report(
        themes,
        clustered_questions,
        clustered_responses,
        all_qa
    )

    # 7. Evaluate whether responses resolve questions
    final_report = evaluate_response_resolution(llm_client, final_report)

    # 8. Compute theme-level resolution scores
    final_report = compute_theme_resolved_scores(final_report)

    # 9. Save final report
    with open("outputs/final_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    # 10. Build and save per-call analysis (NEW artifact)
    transcripts = df.to_dict("records")

    per_call_data = build_per_call_analysis(transcripts, clustered_questions)
    save_per_call_analysis(per_call_data)

    # 11. Compute per-call resolution score (quick ranking preview)
    calls = json.load(open("outputs/per_call_analysis.json"))

    for c in calls:
        scores = [q["question_resolved_score"] for q in c["questions"]]
        c["call_resolution_score"] = sum(scores) / len(scores) if scores else 1

    worst_calls = sorted(calls, key=lambda x: x["call_resolution_score"])[0:20]

    with open("outputs/worst_calls_preview.json", "w") as f:
        json.dump(worst_calls, f, indent=2)

    # 12. Save flattened parquet for analytics
    pd.json_normalize(
        per_call_data,
        record_path="questions",
        meta=["call_id", "agent_id"]
    ).to_parquet("outputs/per_call_analysis.parquet")

    print("Pipeline complete. Outputs written to /outputs")
