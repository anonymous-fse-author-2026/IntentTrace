# IntentTrace: Traceable Diagnosis of LLM-Generated Process Models Against Natural-Language Specifications

## Abstract

Large language models (LLMs) are increasingly used to generate behavioural process models, expressed in BPMN or as UML activity diagrams, from natural-language specifications. Generated models, however, may be malformed or fail to preserve the specification's intent and therefore require refinement. Recent studies show that  accurate defect detection is a main bottleneck in LLM-based refinement, and effective diagnostics are needed to identify defects and trace them to the relevant parts of the specification and model. 
We present IntentTrace, an automated method for detecting defects in LLM-generated process models, both in their well-formedness and in their alignment with the specification, i.e., whether they preserve its intent. IntentTrace restricts the LLM to interpreting natural language and performs the remaining tasks algorithmically: checking structural and logical validity, and comparing the executions described by the specification with those of the model using a trace-differencing algorithm. The diagnostics require no ground-truth model and are traceable to specification spans and model elements, enabling their correctness to be checked. IntentTrace further provides a ground-truth-free specification-alignment measure that agrees with existing ground-truth-based measures nearly as strongly as these measures agree with each other. We evaluate IntentTrace on 700 LLM-generated models from industrial and public specifications. We show that, with a freely available LLM (Qwen~3.6~27B), refinement guided by IntentTrace's diagnostics significantly outperforms refinement guided by state-of-the-art LLM-based critiques according to both ground-truth-based measures and our specification-alignment measure. With a proprietary LLM (Gemini~3.1~Pro), IntentTrace-guided refinement significantly outperforms LLM-based critiques according to our specification-alignment measure, while ground-truth-based measures show only small differences in either direction. Finally, a systematic human evaluation shows that, for 75.7% of the sampled models, every IntentTrace diagnostic identifies a genuine failure to preserve the specification's intent.

## Setup

Install the dependencies with:

`pip install -r requirements.txt`

The LLM interface is implemented in `IntentTrace/llm.py`, which uses [LiteLLM](https://github.com/BerriAI/litellm). The supported models are declared in the `MODELS` dictionary and can be selected by name (e.g. `qwen`, `gemini`). To use a different LLM, add an entry to `MODELS` or override the `LLMCaller` implementation. Credentials are read from the environment (a `.env` file placed in the `IntentTrace` directory is loaded automatically).

## Getting Started

The core method is implemented in the `IntentTrace` directory and is invoked as a module:

`python -m IntentTrace [MODEL_FILE] [TEXT_FILE] --model [MODEL_NAME] --name [OUTPUT_BASE] --verbose`

where `MODEL_FILE` is the process model expressed in our DSL, and `TEXT_FILE` is the natural-language specification the model is diagnosed against. Additional flags:

* `--no-var-actor` disables the variable and actor validation (SMC6/SMC7).
* `--spec` / `--spec-traces` reuse a previously computed specification segmentation and its execution traces instead of recomputing them.

A single run writes a set of files sharing the output base name:

| Extension | Content |
| --- | --- |
| `.grammar` | Grammar-conformance report for the input model (accepted/rejected statements). |
| `.json` | Parsed process model together with the segmented specification elements. |
| `.constraints` | Results of all structural (STC) and logical-sanity (LSC) checks. |
| `.actorvars` | Variable-definition (SMC6) and actor-grouping (SMC7) validation. |
| `.mapping` | Traceability mapping between specification spans and model elements (matches, mutations, omissions). |
| `.trace` | Model execution traces and specification execution traces. |
| `.evaluation` | Trace-differencing results and counter-examples. |
| `.feedback` | Consolidated, human-readable diagnostics used to guide refinement. |
| `.metrics` | Specification-alignment measure and issue counts. |
| `.puml` | PlantUML rendering of the model. |
| `.cost` | Per-call LLM cost, tokens, and durations. |

### Components

* `grammar.py` - the DSL parser; only grammar-conforming statements enter the diagnosis.
* `constraint_checker.py` - the algorithmic structural (STC1–STC22) and logical-sanity (LSC1–LSC3) constraints; see `SupplementaryMaterials/Constraints.pdf`.
* `smt.py` - SMT-based (Z3) reasoning used for the logical-sanity checks.
* `condition.py` / `parsed_model.py` - guard-condition parsing and the internal model representation.
* `element_matching.py` - LLM-based segmentation of the specification and matching of spans to model elements.
* `trace_extraction.py` - extraction of the execution traces described by the specification, and enumeration of the model's init-to-end traces.
* `actor_var_verification.py` - variable and actor validation.
* `evaluation.py` - trace differencing, counter-example generation, consolidated feedback, and the specification-alignment measure.
* `refinement.py` - the refinement loop that repairs models from IntentTrace diagnostics.

### Refinement

To run the refinement loop over a set of results:

`python -m IntentTrace.refinement --dataset [PAGED|Industry] --llm [RUN_FOLDER|FAMILY] --model [gemini|qwen] --overwrite`

Omitting `--dataset` or `--llm` processes all datasets and all run folders. Each refined model is written as a `.refined` file alongside a re-validation report, so the refined model can be diagnosed again in the next round. the cost of refinement is stored in a `.cost_refinement` file.

## Complete Prompts

All prompts are provided in the `IntentTrace/prompts` directory:

* `element_matching.py` - specification segmentation and element matching.
* `trace_extraction.py` - specification-trace extraction.
* `var_actor_matching.py` - variable and actor validation.
* `refinement.py` - refinement prompts for the IntentTrace and ablated variants.
* `direct_llm_feedback.py` - prompt for the Direct-LLM critique baseline.

## Baselines and Ablations

The `Baseline` directory contains the the two ablations, the baseline and the ground-truth-based measures.

* `DirectLLM/direct_llm_feedback.py` generates the **Direct-LLM** critique baseline, in which the LLM is asked directly for a critique of the model.

  `python Baseline/DirectLLM/direct_llm_feedback.py --dataset [PAGED|Industry] --model [RUN_FOLDER] --family [gemini|qwen]`
* `Ablation/generate_feedback.py` generates the ablated variants of IntentTrace from the Round0 models: **Model-Internal** (well-formedness diagnostics only) and **Element-Alignment** (element-level alignment without trace differencing), as well as the **Direct-LLM** baseline.

  `python Baseline/Ablation/generate_feedback.py --variant [VARIANT] --dataset [DATASET] --model [RUN_FOLDER]`

### Ground-Truth-Based Measures

* `Metrics/abscon_eval.py` implements the **AbsCon** soft-cardinality relation-based measure.
* `Metrics/bmatch.py` implements the **LADEX** (B-Match) behavioural node-matching measure.
* `Metrics/build_process_model.py` builds the graph representation the measures operate on.
* `Metrics/add_baseline_scores.py` computes both measures for the generated models and writes the scores back into the per-diagram `.metrics` files.

  `python Baseline/Metrics/add_baseline_scores.py --round [ROUND] --approach [APPROACH] --dataset [DATASET] --metric [both|abscon|ladex]`

## Rendering and Conversion

The `Render` directory converts between representations:

* `dsl_to_csv.py` - DSL to the intermediate CSV representation: `python Render/dsl_to_csv.py --input [FILE_OR_DIR] --output [FILE_OR_DIR]`
* `csv_to_dsl.py` - CSV back to the DSL: `python Render/csv_to_dsl.py --input [FILE_OR_DIR] --output [FILE_OR_DIR]`
* `csv_to_plantuml.py` - CSV to a **PlantUML** diagram for human-oriented visualization.
* `dsl_to_xmi.py` - DSL (or a `.json` parsed model) to a standard-compliant **UML 2.5 XMI** serialization of the MOF-based UML metamodel, for use in UML tooling such as Eclipse Papyrus: `python Render/dsl_to_xmi.py --input [FILE_OR_DIR] --output [FILE_OR_DIR]`

The DSL elements and grammar is documented in `SupplementaryMaterials/DSL-Grammar.pdf`.

## Datasets

The `Datasets` directory contains:

* `All/PAGED` - the full **PAGED** dataset: `Original` contains the process descriptions and `DSL` contains the corresponding models.
* `Industry_Anonymized_Dataset` - a subset of the **Industry** dataset: `PD` contains anonymized process descriptions and `PM` the anonymized ground-truth models.
* `Sampled/PAGED` and `Sampled/Industry` - the sampled subsets used in the evaluation, each with the ground truth (`GT`), the generated models, and their CSV/DSL forms. `mapping.csv` records the mapping from sample ids to source diagrams.
* `All/sample_diagrams.py` and `All/stratification.csv` - the sampling procedure and the resulting stratification.

## Evaluation Results

All results are in the `Results` directory, organized as `Results/Round[N]/[APPROACH]/[DATASET]/[LLM-RUN]/[DIAGRAM_ID]`.

* `Round0` contains the diagnostics computed on the initially generated models, for **IntentTrace** and for the three comparison approaches (**Direct-LLM**, **Model-Internal**, **Element-Alignment**).
* `Round1` contains the models refined from those diagnostics, together with their re-diagnosis.
* `Round2`–`Round4` contain the subsequent IntentTrace refinement rounds.
* Each diagram directory contains the full set of output files described above; the `.metrics` file holds the specification-alignment measure together with the AbsCon and LADEX scores.

## Research Questions (RQ1, RQ2, RQ3)

Results for the research questions are located in the `RQ1`, `RQ2`, and `RQ3` directories. Each script writes its table as a `.csv` file (and, where applicable, a `.pdf`/`.svg` figure) next to itself.

### RQ1

`RQ1/RQ1_Rounds` - effectiveness of IntentTrace-guided refinement across rounds:

* `rq1_quality.py` - precision, recall, and F1 per dataset, LLM, and round.
* `rq1_issues.py` - number of diagnosed issues per round.
* `rq1_cost.py` - LLM calls, tokens, and cost.
* `rq1_significance.py` and `rq1_baseline_significance.py` - Wilcoxon paired tests and A12 effect sizes between rounds and against the baselines, reported for the specification-alignment measure and for AbsCon and LADEX.
* `rq1_cross_model.py` - comparison across LLM families.
* `results_reader.py`, `rq1_data.py`, `table_common.py`, `sig_table.py` - result loading and table formatting helpers.

`RQ1/RQ1_Ablation` - the same analyses for the ablated variants and the Direct-LLM baseline (`rq1_ablation_quality.py`, `rq1_ablation_spec_quality.py`, `rq1_ablation_cost.py`, `rq1_ablation_significance.py`, `rq1_ablation_spec_significance.py`, `rq1_ablation_baseline_*`).

`RQ1/report_common.py` and `RQ1/sig_common.py` provide the shared aggregation and statistical-testing code.

### RQ2

Human evaluation of the traceable diagnostics.

* `Annotator1` and `Annotator2` contain the per-diagram material shown to each annotator and their `annotation_sheet.xlsx`.
* `stratification.py` produces the sampling used for the human study (`stratification.csv`, `manifest.csv`).
* `agreement.py` computes inter-annotator agreement and writes `agreement_results.csv`.
* `consensus.xlsx` contains the consensus labels after resolving disagreements.

### RQ3

Agreement between the ground-truth-free specification-alignment measure and the ground-truth-based measures.

* `rq3_baseline_consistency.py` computes the Spearman correlations between the specification-alignment measure, AbsCon, and LADEX, and writes `rq3_baseline_consistency.csv`.

## Supplementary Materials

The `SupplementaryMaterials` directory contains:

* `DSL-Grammar.pdf` - the full grammar of the process-model DSL.
* `Constraints.pdf` - all structural and logical-sanity constraints.
* `AlgorithmExplanation.pdf` - a detailed explanation of the trace-differencing algorithm stated in the paper.
* `PropositionProof.pdf` - the proof of the proposition stated in the paper.
* `RoundByRoundResults.pdf` - complete round-by-round results.
* `AblationBaselineResults.pdf` - complete ablation and baseline results.
