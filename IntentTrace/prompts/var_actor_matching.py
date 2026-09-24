from typing import Optional

VAR_ACTOR_MATCHING_CORE = """# ROLE
You audit the variable and actor declarations of a generated process model against the source process text. The declarations arrive as already-parsed JSON records, not raw syntax. Sort every record into one of four buckets: match, mutation, omission, or addition.

# SCOPE
You judge only these records:
- variable: `{ "ref", "id", "name", "range" }` — does its name and value space say what the text says?
- actor:    `{ "ref", "id", "name", "assigned_nodes" }` — does it do what the text says this participant does?

A node list will be provided with them for looking up assigned node ids. Do not classify nodes, transitions, labels, or guards; other steps check those. 

# VARIABLES
Range notation: `{[-]}` any number, `{[a-]}` at least a, `{[-b]}` at most b, `{[a-b]}` the closed interval, `{x, y, z}` a finite set of categories.

- A range is the value space, not the condition. If the text gives only a threshold ("above 80"), the correct range is `{[-]}`.
- If the text gives explicit bounds or explicit categories, the range must match them.
- The name must name the quantity, as a noun phrase. A paraphrase of it is a match; a name that folds the condition in ("ScoreOver1") is a mutation.
- The same variable may serve several guards or branches, so do not expect one variable per branch.

# ACTORS
- A paraphrased actor name is a match as long as the role survives.
- If the text names no participants at all, a single actor owning every node is a match.
- Before anything else, decide whether the actor's role appears in the source. If it does not, it is an ADDITION even when it holds nodes — never a mutation.
- When a node sits with the wrong participant and both participants appear in the source, report TWO mutations, naming the node id in each: one on the actor wrongly holding it, and one on the actor that should hold it but whose `assigned_nodes` now lacks it.
- When the actor wrongly holding it is not in the source, report that actor as an ADDITION instead, plus the mutation on the actor now missing the node.

# THE FOUR BUCKETS
1. **MATCH** — the declaration agrees with the source.
2. **MUTATION** — the declaration maps to the source but its payload is wrong: a changed range, a name encoding a condition, a wrong actor name, or a node assigned to the wrong participant.
3. **OMISSION** — the source calls for a declaration that is absent.
4. **ADDITION** — the declaration has no support in the source.

Every record you are given must land in exactly one of matches, mutations, or additions.
"""

VAR_ACTOR_MATCHING_EXAMPLES = """# FEW-SHOT EXAMPLES

### Example 1 — threshold-only range is correct; no named participants
**Process Description:**
"If the package weight is above 80, request an inspection; otherwise route it to dispatch."

**Node List (for context):**
[
    {"id": "e1", "kind": "init"},
    {"id": "e2", "kind": "action", "content": "Read package weight"},
    {"id": "e3", "kind": "branch", "gateway": "XOR"},
    {"id": "e4", "kind": "action", "content": "Request inspection"},
    {"id": "e5", "kind": "action", "content": "Route to dispatch"},
    {"id": "e6", "kind": "end"}
]

**Variable Records under audit:**
[
    {"ref": "variable:v1", "id": "v1", "name": "PackageWeight", "range": "{[-]}"}
]

**Actor Records under audit:**
[
    {"ref": "actor:a1", "id": "a1", "name": "System", "assigned_nodes": ["e1","e2","e3","e4","e5","e6"]}
]

**Output:**
{
    "variables": {
        "matches": [
            {
                "refs": ["variable:v1"],
                "justification": "The source gives only a threshold, so the unbounded range is the right value space, and the name identifies the quantity."
            }
        ],
        "mutations": [],
        "omissions": [],
        "additions": []
    },
    "actors": {
        "matches": [
            {
                "refs": ["actor:a1"],
                "justification": "The source names no participants, so one actor covering every node is acceptable."
            }
        ],
        "mutations": [],
        "omissions": [],
        "additions": []
    }
}

### Example 2 — mismatched interval and category set; misassigned node; missing actor
**Process Description:**
"Continue only when the temperature stays between 5 and 15 degrees. The auditor verifies the report; the operator files it. The clerk later archives a copy."

**Node List (for context):**
[
    {"id": "e1", "kind": "init"},
    {"id": "e2", "kind": "action", "content": "Verify report"},
    {"id": "e3", "kind": "action", "content": "File report"},
    {"id": "e4", "kind": "action", "content": "Archive copy"},
    {"id": "e5", "kind": "end"}
]

**Variable Records under audit:**
[
    {"ref": "variable:v1", "id": "v1", "name": "Temperature", "range": "{[0-20]}"},
    {"ref": "variable:v2", "id": "v2", "name": "AuditScore", "range": "{[-]}"}
]

**Actor Records under audit:**
[
    {"ref": "actor:a1", "id": "a1", "name": "Auditor", "assigned_nodes": ["e1","e2","e3"]},
    {"ref": "actor:a2", "id": "a2", "name": "Operator", "assigned_nodes": ["e4","e5"]}
]

**Output:**
{
    "variables": {
        "matches": [],
        "mutations": [
            {
                "refs": ["variable:v1"],
                "reason": "The source states the closed interval [5, 15]; the declared range [0, 20] disagrees. The name is correct."
            }
        ],
        "omissions": [],
        "additions": [
            {
                "refs": ["variable:v2"],
                "reason": "The source describes no audit score, and no condition needs such a variable."
            }
        ]
    },
    "actors": {
        "matches": [],
        "mutations": [
            {
                "refs": ["actor:a1"],
                "reason": "Filing the report (e3) belongs to the Operator in the source, not the Auditor, so e3 is wrongly held here."
            },
            {
                "refs": ["actor:a2"],
                "reason": "The source has the Operator file the report (e3), but e3 is missing from this actor's assigned_nodes."
            }
        ],
        "omissions": [
            {
                "summary": "Clerk role",
                "reason": "The source has the clerk archive a copy, but no actor declaration covers that role."
            }
        ],
        "additions": []
    }
}
"""

VAR_ACTOR_MATCHING_INPUTS = """# INPUT DATA

**Process Description:**
{{process_text}}

**Node List (id, kind, content — for actor-assignment context):**
{{node_catalog}}

**Variable Records under audit:**
{{variable_declarations}}

**Actor Records under audit:**
{{actor_declarations}}
"""

VAR_ACTOR_MATCHING_OUTPUT = """# OUTPUT FORMAT
Respond with a raw JSON object. No Markdown fences, no text outside the JSON. Copy each record's `ref` string exactly into `refs`, and return an empty list for any bucket with no findings. For an omission, give a short `summary` of what is missing plus a one-line `reason`. Use exactly these keys:

{
    "variables": {
        "matches": [
            {
                "refs": ["<variable ref>"],
                "justification": "<why the declaration matches the source>"
            }
        ],
        "mutations": [
            {
                "refs": ["<variable ref>"],
                "reason": "<what its payload gets wrong, and what the source says>"
            }
        ],
        "omissions": [
            {
                "summary": "<short label for the missing variable or value space>",
                "reason": "<why the source calls for a declaration that is absent>"
            }
        ],
        "additions": [
            {
                "refs": ["<variable ref>"],
                "reason": "<why this declaration has no source support>"
            }
        ]
    },
    "actors": {
        "matches": [
            {
                "refs": ["<actor ref>"],
                "justification": "<why the declaration matches the source>"
            }
        ],
        "mutations": [
            {
                "refs": ["<actor ref>"],
                "reason": "<what its payload gets wrong: a wrong name, or a node held by the wrong participant>"
            }
        ],
        "omissions": [
            {
                "summary": "<short label for the missing actor or responsibility>",
                "reason": "<why the source calls for a declaration that is absent>"
            }
        ],
        "additions": [
            {
                "refs": ["<actor ref>"],
                "reason": "<why this declaration has no source support>"
            }
        ]
    }
}
"""


def build_var_actor_matching_prompt(
    main_prompt: Optional[str] = None,
    examples: Optional[str] = None,
    inputs_template: Optional[str] = None,
    output_format: Optional[str] = None,
) -> str:
    main = main_prompt or VAR_ACTOR_MATCHING_CORE
    ex = examples or VAR_ACTOR_MATCHING_EXAMPLES
    inp = inputs_template or VAR_ACTOR_MATCHING_INPUTS
    out = output_format or VAR_ACTOR_MATCHING_OUTPUT

    return f"""{main}
---

{ex}

---

{inp}

---

{out}
"""


VAR_ACTOR_MATCHING_PROMPT = build_var_actor_matching_prompt()
