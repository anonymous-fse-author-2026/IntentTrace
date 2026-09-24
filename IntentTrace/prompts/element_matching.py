ELEMENT_MATCHING_CORE = """# ROLE
You audit a process model against the source text it was built from. The model arrives as a list of already-parsed elements (structured JSON, not raw syntax). For each element, find the exact source spans that justify it, and sort every element and every described step into one of four buckets: match, mutation, omission, or addition.

# SCOPE
Each element is a JSON object with a stable `ref` and a `kind`:
    - action      one step the process performs (carries `content`)
    - guard       a condition on a connection (carries `content`, plus `from`/`to`)
    - label       descriptive text on a connection (carries `content`, plus `from`/`to`)
    - init        the start of the process
    - end         a finishing point
    - branch      a split in the flow (carries a `gateway` of AND, OR, or XOR)
    - converge    a point where split flows rejoin

Only `action`, `label`, and `guard` have explicit content. The source often implies the other four rather than naming them.

You judge the steps and the flow only:
- Every step and decision in the text appears in the model, and the model adds nothing the text does not describe.
- Each element keeps the meaning, values, and units of the text it came from.
- Splits, choices, and joins reflect the logic the text implies.

You will NOT receive precedence edges, variable definitions, or actor assignments. Other steps check those, so do not invent them or reason about them here.

# GROUPING AN ENTRY
The `refs` and `verbatim_texts` arrays let one entry carry several of either:
- Several elements, one span. A span often justifies more than one element: "Start when the order arrives" justifies both `init` and an action "Order arrives"; "If the total exceeds 5000, escalate" justifies both an XOR `branch` and the action "Escalate". List every element it supports in that one entry. Sharing a span is not a reason to report any of them as a defect.
- One element, several spans. A single element is often spread across the text: an action "Record and archive the claim" is justified by "records it" plus "the claim is archived". List every span that supports it in that one entry.

Never put several elements and several spans in the same entry — split those into separate entries.

# EXTRACTING SPANS
- Copy span texts verbatim: never extract both a phrase and a larger phrase containing it.
- If the text states a condition, keep the "if"/"when" marker with its subject ("If the total invoice amount"), and extract the comparison and value as their own span ("is less than 4").
- If the text describes steps running at the same time ("simultaneously", "concurrently", etc.), extract that wording.
- If the text only orders sequential steps ("Following this", "then", "afterwards"), do not extract it. Leaving it out is not an omission.
- If a span joins parts with "and"/"or", split them, keeping each part's subject or context.
- If a span describes branches converging ("once tasks are finished", "when selected"), extract that wording.
- If a lead-in only announces that steps follow ("the system does two things", "There are two cases:"), do not extract it. Leaving it out is not an omission.
- Every word saying who, what, or how much must appear in some span.

# THE CATEGORIZATION
Preserved meaning is a MATCH. Tense, word form, and phrasing may differ freely. Only drifted meaning is a MUTATION — changed logic, reversed outcome, different target, or a changed operator, value, range, unit, or count. Be strict. Judge an element's content against the source, never how the modeller chose to represent it. An element that repeats or restates something already modeled is acceptable.

1. **MATCH** — one or more spans support the element with the same meaning and values. For guards, the operator and the value must both agree. An `init`, `end`, `branch`, or `converge` that a span merely implies belongs here too.
2. **MUTATION** — the element maps to the source, but its payload or logic is wrong (`>` where the text says `>=`, a changed threshold, unit, or count, a reversed outcome, a wrong branch target).
3. **OMISSION** — the source clearly describes a step or decision that no element implements. Point at the missing step or split, never at an announcer. Raise a missing `init`, `end`, `branch`, or `converge` only where the text explicitly calls for it.
4. **ADDITION** — the element has no source support at all.
"""
ELEMENT_MATCHING_EXAMPLES = """# FEW-SHOT EXAMPLES

### Example 1 — all four buckets; a shared span; an unjustified `end`
**Raw Input Text:**
"When a user signs up, validate their email format. If the age is greater than or equal to 18, grant access to the dashboard. Otherwise, reject the user. Finally, send a welcome email."

**Generated Model Elements:**
[
    {"ref": "init:e1", "kind": "init"},
    {"ref": "action:e2", "kind": "action", "content": "Validate email format"},
    {"ref": "branch:e3", "kind": "branch", "gateway": "XOR"},
    {"ref": "guard:e3->e4", "kind": "guard", "content": "v1 >= 21", "from": "e3", "to": "e4"},
    {"ref": "action:e4", "kind": "action", "content": "Grant access"},
    {"ref": "action:e5", "kind": "action", "content": "Reject user"},
    {"ref": "action:e6", "kind": "action", "content": "Initialize database"},
    {"ref": "end:e7", "kind": "end"}
]

**Output:**
{
    "matches": [
        {
            "refs": ["init:e1", "action:e2"],
            "verbatim_texts": ["When a user signs up, validate their email format"],
            "justification": "The opening clause implicitly starts the process AND describes the first action."
        },
        {
            "refs": ["branch:e3"],
            "verbatim_texts": ["If the age is greater than or equal to 18"],
            "justification": "The conditional implies the XOR decision."
        },
        {
            "refs": ["action:e4"],
            "verbatim_texts": ["grant access to the dashboard"],
            "justification": "Paraphrase of the positive arm."
        },
        {
            "refs": ["action:e5"],
            "verbatim_texts": ["Otherwise, reject the user"],
            "justification": "Paraphrase of the negative arm."
        }
    ],
    "mutations": [
        {
            "refs": ["guard:e3->e4"],
            "verbatim_texts": ["is greater than or equal to 18"],
            "reason": "The guard threshold is 21 but the source says 18."
        }
    ],
    "omissions": [
        {
            "verbatim_texts": ["Finally, send a welcome email."],
            "reason": "No element implements this final step."
        }
    ],
    "additions": [
        {
            "refs": ["action:e6"],
            "reason": "No source text supports this action."
        }
    ]
}
Notes:
- `init:e1` shares an entry with `action:e2` because one span justifies both. Do NOT demote `init:e1` to an omission because the action absorbed the span.
- No span justifies `end:e7`. Since an `end` needs none, it is simply left out of the output — no ADDITION and no OMISSION for it.

### Example 2 — extracting a condition as branch, guard, and action spans; a loose label
**Raw Input Text:**
"If the parcel weight is over 50, route it to freight; otherwise use standard post. Once the parcel is labelled for dispatch, hand it to the courier."

**Generated Model Elements:**
[
    {"ref": "branch:e1", "kind": "branch", "gateway": "XOR"},
    {"ref": "guard:e1->e2", "kind": "guard", "content": "v1 > 50", "from": "e1", "to": "e2"},
    {"ref": "action:e2", "kind": "action", "content": "Route to freight"},
    {"ref": "action:e3", "kind": "action", "content": "Use standard post"},
    {"ref": "label:e4->e5", "kind": "label", "content": "parcel labelled", "from": "e4", "to": "e5"},
    {"ref": "action:e5", "kind": "action", "content": "Hand to courier"}
]

**Output:**
{
    "matches": [
        {
            "refs": ["branch:e1"],
            "verbatim_texts": ["If the parcel weight"],
            "justification": "Marker plus subject; the conditional implies the XOR split."
        },
        {
            "refs": ["guard:e1->e2"],
            "verbatim_texts": ["is over 50"],
            "justification": "Operator and value extracted as their own span; both agree with the guard."
        },
        {
            "refs": ["action:e2"],
            "verbatim_texts": ["route it to freight"],
            "justification": "Paraphrase of the positive arm."
        },
        {
            "refs": ["action:e3"],
            "verbatim_texts": ["otherwise use standard post"],
            "justification": "Paraphrase of the negative arm."
        },
        {
            "refs": ["label:e4->e5"],
            "verbatim_texts": ["the parcel is labelled for dispatch"],
            "justification": "The label paraphrases this span; the bare ordering word 'Once' is left out."
        },
        {
            "refs": ["action:e5"],
            "verbatim_texts": ["hand it to the courier"],
            "justification": "Paraphrase of the final step."
        }
    ],
    "mutations": [],
    "omissions": [],
    "additions": []
}

### Example 3 — an implied converge, and a condition modeled as an action
**Raw Input Text:**
"The two inspections finish and the supervisor signs the report. If the total exceeds 5000, trigger a manual review."

**Generated Model Elements:**
[
    {"ref": "converge:e1", "kind": "converge"},
    {"ref": "action:e2", "kind": "action", "content": "Sign report"},
    {"ref": "action:e3", "kind": "action", "content": "Check whether the total exceeds 5000"},
    {"ref": "action:e4", "kind": "action", "content": "Trigger a manual review"}
]

**Output:**
{
    "matches": [
        {
            "refs": ["converge:e1", "action:e2"],
            "verbatim_texts": ["The two inspections finish and the supervisor signs the report."],
            "justification": "Completing the parallel inspections implies the converge, and the same span describes the sign-report action that follows."
        },
        {
            "refs": ["action:e3"],
            "verbatim_texts": ["If the total exceeds 5000"],
            "justification": "The condition is modeled as an action step, but its content faithfully restates the source, so it is a MATCH rather than an addition or mutation."
        },
        {
            "refs": ["action:e4"],
            "verbatim_texts": ["trigger a manual review"],
            "justification": "Direct paraphrase."
        }
    ],
    "mutations": [],
    "omissions": [],
    "additions": []
}
"""
ELEMENT_MATCHING_INPUTS = """# INPUT DATA FOR MATCHING

**Raw Input Text:**
{{input_text}}

**Generated Model Elements:**
{{elements}}
"""
ELEMENT_MATCHING_OUTPUT = """# OUTPUT FORMAT
Respond with a raw JSON object. No Markdown fences, no text outside the JSON. Copy each element's `ref` string exactly into `refs`. Return an empty list for any bucket with no findings.
Use exactly these keys:

{
    "matches": [
        {
            "refs": ["<one or more element `ref` values>"],
            "verbatim_texts": ["<one or more exact copy-paste spans from the Input Text>"],
            "justification": "<why the element and the spans align>"
        }
    ],
    "mutations": [
        {
            "refs": ["<one or more element `ref` values>"],
            "verbatim_texts": ["<one or more exact copy-paste spans from the Input Text>"],
            "reason": "<which payload, value, or logic is wrong, and what the source says>"
        }
    ],
    "omissions": [
        {
            "verbatim_texts": ["<one or more exact copy-paste spans the model never implements>"],
            "reason": "<what the model failed to implement>"
        }
    ],
    "additions": [
        {
            "refs": ["<one or more element `ref` values>"],
            "reason": "<why these elements have no source support>"
        }
    ]
}
"""

def build_element_matching_prompt(main_prompt=None, examples=None, inputs_template=None, output_format=None):
    main = main_prompt or ELEMENT_MATCHING_CORE
    ex = examples or ELEMENT_MATCHING_EXAMPLES
    inp = inputs_template or ELEMENT_MATCHING_INPUTS
    out = output_format or ELEMENT_MATCHING_OUTPUT

    return f"""{main}
---

{ex}

---

{inp}

---

{out}
"""

ELEMENT_MATCHING_PROMPT = build_element_matching_prompt()


ELEMENT_MATCHING_SPEC_INPUTS = """# INPUT DATA FOR MATCHING

**Raw Input Text:**
{{input_text}}

**Spec Elements (the source text already segmented into stable units):**
{{spec_elements}}

**Spec Traces (execution paths through the spec elements):**
{{spec_traces}}

**Generated Model Elements:**
{{elements}}
"""

ELEMENT_MATCHING_SPEC_OUTPUT = """# WORKING FROM THE SUPPLIED SPEC
The source text has already been segmented into the Spec Elements above, and the Spec Traces are the execution paths through them. Do not segment the source text again.

Reuse that spec. Keep each element's id and `content` exactly as given, and change it only where it is genuinely wrong:
- Edit a `content` when the span was never in the source, when it severs a condition from its subject or an operator from its value, or when it swallowed several steps the model rightly keeps apart. Keep the element's id when you edit it.
- Add an element only for source text carrying behaviour that no existing element covers, using the next unused id. Extract it the same way as described above.
- Remove an element only when the source does not support it, or when it is a pure announcer.
- Never renumber. An id keeps pointing at the same piece of source text, and the id of a removed element is not reused.

Update the traces only to stay consistent with the elements you return: drop removed ids, thread added ids in where the source puts them, and drop a trace the source no longer supports. Every id in a trace must exist in the elements you return.

Judge the model against the spec you return, and fill `verbatim_texts` by copying the `content` of the relevant spec element exactly, character for character.

# OUTPUT FORMAT
Respond with a raw JSON object. No Markdown fences, no text outside the JSON. Use exactly these keys. List only actual changes in `spec_revisions`; leave it empty when you kept the spec as it was:

{
    "spec_elements": [
        {"id": "<spec id>", "content": "<exact source span>"}
    ],
    "spec_traces": [
        ["<spec id>", "<spec id>", "..."]
    ],
    "spec_revisions": [
        {
            "action": "edited | added | removed",
            "ids": ["<spec ids this entry covers>"],
            "reason": "<what was wrong and how you fixed it>"
        }
    ],
    "matches": [
        {
            "refs": ["<one or more element `ref` values>"],
            "verbatim_texts": ["<exact `content` of the supporting spec element(s)>"],
            "justification": "<why the element and the spans align>"
        }
    ],
    "mutations": [
        {
            "refs": ["<one or more element `ref` values>"],
            "verbatim_texts": ["<exact `content` of the distorted spec element(s)>"],
            "reason": "<which payload, value, or logic is wrong, and what the source says>"
        }
    ],
    "omissions": [
        {
            "verbatim_texts": ["<exact `content` of the spec element(s) nothing implements>"],
            "reason": "<what the model failed to implement>"
        }
    ],
    "additions": [
        {
            "refs": ["<one or more element `ref` values>"],
            "reason": "<why these elements have no source support>"
        }
    ]
}
"""

ELEMENT_MATCHING_SPEC_PROMPT = build_element_matching_prompt(
    inputs_template=ELEMENT_MATCHING_SPEC_INPUTS,
    output_format=ELEMENT_MATCHING_SPEC_OUTPUT,
)
