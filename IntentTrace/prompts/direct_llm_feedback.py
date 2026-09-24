from .refinement import REFINEMENT_DSL_GRAMMAR

DIRECT_LLM_FEEDBACK_CONSTRAINTS = """Semantic Constraints (SMC):
- SMC1: no omitted or hallucinated documented behaviors.
- SMC2: node payload meaning is faithful to source text.
- SMC3: precedence preserves source chronology exactly.
- SMC4: control-flow logic (choices, concurrency, cycles, joins) is correctly represented.
- SMC5: each init-to-end path mirrors an intended source progression.
- SMC6: variables and ranges faithfully reflect the source.
- SMC7: actor assignments faithfully reflect responsibilities in the source."""


def build_direct_llm_feedback_prompt(process_description: str, dsl_model: str) -> str:
    return (
        "You review behavioral process models. You are given a process description "
        "and a model of it written in a DSL. Report, precisely and concisely, where "
        "the model does not faithfully reflect the description.\n\n"
        "### 1. DSL Grammar\n"
        f"{REFINEMENT_DSL_GRAMMAR}\n\n"
        "### 2. Semantic Constraints to Judge Against\n"
        "Judge the model only against the constraints below.\n"
        f"{DIRECT_LLM_FEEDBACK_CONSTRAINTS}\n\n"
        "### 3. Process Description\n"
        f"{process_description}\n\n"
        "### 4. Process Model (the DSL to review)\n"
        f"{dsl_model}\n\n"
        "--- INSTRUCTIONS ---\n"
        "Report only the discrepancies you are confident about, one bullet each, as "
        "'- <element or text>: <what is wrong>'. Cite concrete DSL element ids (e.g. "
        "action(n5,...), guard(b2,n7,...)) and quote the source text you are judging "
        "against. Be specific and strict, and do not speculate: list only what the "
        "description clearly implies. If the model is semantically faithful, output "
        "exactly 'No semantic issues found.'\n"
        "Output ONLY the findings, with no preamble, reasoning, or summary."
    )
