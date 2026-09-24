TRACE_EXTRACTION_CORE = """# ROLE
You read a process and list its distinct execution paths. Each path is the ordered sequence of step IDs the process follows from start to finish. Together the paths must cover every route the process can take from the start to an end, with no route invented and none left out.

# INPUT
Each step is given as `[ID: X] <text>`, with alphanumeric IDs (`i1`, `i2`, ...). Work out the start, the decisions, the loops, and the end points from the step text and the overall meaning.

# RULES

1. **Keep IDs exact.** Use the IDs exactly as given: never renumber them, strip prefixes, or invent new ones.

2. **One start.** Exactly one step starts the process, and every path begins with that same ID.

3. **Where a path stops.** A path ends as soon as it either reaches a finishing step or revisits an ID already in that same path. Revisiting closes a cycle: `i1->i2->i3->i2` stops at the second `i2`.

4. **Loops.** Go around a loop once, then stop where it would re-enter. Never unroll a loop into multiple passes. A loop-decision step ("this continues until...") only sends the flow back; the forward exit is at the branch point, not at the loop decision.

5. **Decisions.** Follow every outcome of a decision. An outcome that leads into a loop stops per rule 4; one that continues or exits runs on to its finishing step.

6. **Parallel steps.** Steps that run at the same time each get their own path: if `i3`, `i4`, and `i5` all follow `i2`, produce `...i2->i3->next`, `...i2->i4->next`, and `...i2->i5->next` separately.

7. **Drop contained paths.** If one path's IDs all appear in another in the same order, keep only the longer one.
"""

TRACE_EXTRACTION_EXAMPLES = """# FEW-SHOT EXAMPLES

## Example 1 — a decision whose outcomes loop back and exit
**Original Text:** "Log the request. Inspect it. If it fails inspection, log the request again. If it passes, ship it and close the case."
**Atomic Elements:**
[ID: i1] Log the request
[ID: i2] Inspect it
[ID: i3] If it fails inspection
[ID: i4] If it passes
[ID: i5] ship it
[ID: i6] close the case

**Correct Output:**
[
    ["i1", "i2", "i3", "i1"],
    ["i1", "i2", "i4", "i5", "i6"]
]

## Example 2 — steps running at the same time
**Original Text:** "Open the ticket. Notify the owner and update the dashboard at the same time. Then close the ticket."
**Atomic Elements:**
[ID: i1] Open the ticket
[ID: i2] at the same time
[ID: i3] Notify the owner
[ID: i4] update the dashboard
[ID: i5] close the ticket

**Correct Output:**
[
    ["i1", "i2", "i3", "i5"],
    ["i1", "i2", "i4", "i5"]
]
"""

TRACE_EXTRACTION_INPUTS = """# INPUT DATA
**Original Text:**
{{original_text}}

**Atomic Elements:**
{{atomic_elements}}
"""
TRACE_EXTRACTION_OUTPUT = """# OUTPUT FORMAT
Respond ONLY with a raw JSON array of the execution paths, no markdown formatting and no text outside it.

[
    ["i1", "i2", "i5"],
    ["i1", "i3", "i5"]
]

Every path must begin with the same first ID, since the process has a single start.
"""


def build_trace_extraction_prompt(main_prompt=None, examples=None, inputs_template=None, output_format=None):
    main = main_prompt or TRACE_EXTRACTION_CORE
    ex = examples or TRACE_EXTRACTION_EXAMPLES
    inp = inputs_template or TRACE_EXTRACTION_INPUTS
    out = output_format or TRACE_EXTRACTION_OUTPUT
    
    return f"""{main}
---

{ex}

---

{inp}

---

{out}
"""

TRACE_EXTRACTION_PROMPT = build_trace_extraction_prompt()
