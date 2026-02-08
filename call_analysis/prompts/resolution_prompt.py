RESOLUTION_CHECK_PROMPT = """
You are evaluating whether an agent response successfully resolves a customer's question.

Question:
{question}

Agent Response:
{response}

Definition of RESOLVED:
- Directly answers the question
- Provides actionable next step
- Does not defer without information

Return STRICT JSON:

{
  "resolve_question": 1 or 0,
  "confidence": float between 0 and 1,
  "reasoning": "short explanation"
}
"""
