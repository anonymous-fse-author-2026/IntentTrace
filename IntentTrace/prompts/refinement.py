REFINEMENT_DSL_GRAMMAR = """The model is written in this grammar (EBNF; braces { } are literal):

    <PROCESS>     ::= <DECL_BLOCK> <REL_BLOCK>
    <DECL_BLOCK>  ::= ( <NODE_DECL> | <VAR_DECL> | <ACTOR_DECL> )+
    <NODE_DECL>   ::= ( "init" | "end" | "converge" ) "(" <ID> ")"
                    | "branch" "(" <ID> "," ( "AND" | "OR" | "XOR" ) ")"
                    | "action" "(" <ID> "," <STR> ")"
    <VAR_DECL>    ::= "defines" "(" <VAR> "," <STR> "," <RANGE> ")"
    <ACTOR_DECL>  ::= "actor" "(" <ID> "," <STR> "," "[" <ID> ( "," <ID> )* "]" ")"
    <REL_BLOCK>   ::= ( <PRECEDENCE> | <GUARD> | <LABEL> )+
    <PRECEDENCE>  ::= "precedence" "(" <ID> "," <ID> ")"
    <GUARD>       ::= "guard" "(" <ID> "," <ID> "," <CONDITION> ")"
    <LABEL>       ::= "label" "(" <ID> "," <ID> "," <STR> ")"
    <RANGE>       ::= "{" ( <VAL> ( "," <VAL> )* | "[" <NUM>? "-" <NUM>? "]" )? "}"
    <ID>/<VAR>    ::= <LETTER> ( <LETTER> | <DIGIT> | "_" )*
    <STR>         ::= '"' <CHAR>* '"' | "'" <CHAR>* "'"
    <VAL>         ::= <STR> | <BOOL> | <NUM>
    <BOOL>        ::= "true" | "false"
    <NUM>         ::= "-"? <DIGIT>+ ( "." <DIGIT>+ )?

What each statement means:
- init(id)            : the single start of the process.
- end(id)             : a finishing point (there may be several).
- action(id, "text")  : one step the process performs.
- branch(id, GATEWAY) : a split point; AND runs the outgoing flows in parallel,
                        XOR takes exactly one, OR takes one or more.
- converge(id)        : a point where split flows rejoin.
- defines(v, "name", {range}) : a decision variable and its value space. Ranges:
                        {a, b, c} finite set; {[lo-hi]} numeric interval;
                        {[-]}, {[lo-]}, {[-hi]} unbounded or half-bounded.
- actor(id, "name", [nodes]) : a participant and the node ids it owns.
- precedence(from, to): a directed control-flow edge between two nodes.
- guard(branch, to, condition) : a boolean condition on an outgoing edge of an
                        XOR/OR branch (e.g. v1 = 'Yes', v2 >= 18).
- label(from, to, "text") : descriptive text on an edge."""

REFINEMENT_CONSTRAINTS = """The refined model is re-validated against the rules below, so it must satisfy
all of them — including any a fix might break. Feedback items cite these codes.

Structural Constraints (STC):
- STC1: identifiers are unique/disjoint across nodes, variables, actors.
- STC2: each node has exactly one valid type.
- STC3: exactly one init node.
- STC4: init has no incoming transitions.
- STC5: at least one end node exists.
- STC6: end has no outgoing transitions.
- STC7: action/actor/variable textual declarations are non-empty.
- STC8: each action has exactly one incoming and one outgoing transition.
- STC9: each branch has exactly one incoming and at least two outgoing transitions.
- STC10: each converge has at least two incoming and exactly one outgoing transition.
- STC11: each variable range has at least two distinct values.
- STC12: precedence endpoints must be valid nodes.
- STC13: each guard is attached to an existing transition.
- STC14: each guard expression is syntactically valid.
- STC15: every variable used in a guard is formally defined.
- STC16: each outgoing edge of XOR/OR branches has exactly one guard.
- STC17: guards appear only on outgoing edges of XOR/OR branches.
- STC18: labels appear only on existing transitions.
- STC19: every non-init node is reachable from init.
- STC20: every non-end node can reach an end node.
- STC21: each actor declaration has non-empty assigned responsibilities.
- STC22: each node is assigned to exactly one actor.

Logical Sanity Checks (LSC):
- LSC1: Each guard is satisfiable within its variables' declared domains. This means every condition must actually be possible to trigger based on the range of the variable.
- LSC2: XOR outgoing guards are mutually exclusive. This means no two conditions can ever be true at the exact same time.
- LSC3: XOR/OR outgoing guards are exhaustive over their domains. This means all possible variable values must be covered by at least one path so the system never gets stuck.

Semantic Constraints (SMC) — these underlie the [N*] feedback:
- SMC1: no omitted or hallucinated documented behaviors.
- SMC2: node payload meaning is faithful to source text.
- SMC3: precedence preserves source chronology exactly.
- SMC4: control-flow logic (choices, concurrency, cycles, joins) is correctly represented.
- SMC5: each init-to-end path mirrors an intended source progression.
- SMC6: variables and ranges faithfully reflect the source.
- SMC7: actor assignments faithfully reflect responsibilities in the source."""

def build_refinement_prompt(process_description: str, dsl_model: str, feedback: str,
                 input_elements: str = "None", source_traces: str = "None",
                 model_traces: str = "None") -> str:
    return (
        "You are an expert process model refiner. Your only task is to edit the DSL "
        "process model below so it resolves the given feedback and better matches the "
        "process description.\n\n"
        "### 1. DSL Grammar (the refined model must conform to it)\n"
        f"{REFINEMENT_DSL_GRAMMAR}\n\n"
        "### 2. Constraints\n"
        f"{REFINEMENT_CONSTRAINTS}\n\n"
        "### 3. Source Process\n"
        "3a. Process Description:\n"
        f"{process_description}\n\n"
        "3b. Source-Text Elements:\n"
        f"{input_elements}\n\n"
        "3c. Source-Text Traces (execution paths through the description, by 0-based index):\n"
        f"{source_traces}\n\n"
        "### 4. Current Process Model (the DSL to refine)\n"
        "4a. DSL Process Model:\n"
        f"{dsl_model}\n\n"
        "4b. Model Execution Traces (execution paths through the model, by 0-based index):\n"
        f"{model_traces}\n\n"
        "### 5. Revision Feedback\n"
        "Each item is tagged by what it violates: [S*] a structural rule, [L*] a logical "
        "rule, [N*] faithfulness to the source. Items sometimes cite traces by 0-based "
        "index, e.g. \"input 0\" (Source-Text Trace 0), \"model 1\" (Model Execution "
        "Trace 1), or \"(0->2)\" (source-text trace 0 against model trace 2). Use "
        "sections 3 and 4 to resolve every such reference.\n\n"
        f"{feedback}\n\n"
        "--- REFINEMENT INSTRUCTIONS ---\n"
        "1. Resolve the diagnostics in the listed feedback, one by one, and change nothing else. Do not look "
        "for problems the feedback does not raise. Leave every element it does not "
        "mention exactly as it is — same id, same text, same position in the flow.\n"
        "2. Keep the result conformant to the grammar and the rules above.\n"
        "3. Output ONLY the refined DSL model in a markdown code block, with no "
        "introduction, reasoning, or explanation.\n\n"
    )
