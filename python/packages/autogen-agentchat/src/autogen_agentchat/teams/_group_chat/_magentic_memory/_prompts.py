from pydantic import BaseModel

ORCHESTRATOR_SYSTEM_MESSAGE = ""


ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT = """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""


ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT = """Fantastic. To address this request we have assembled the following team:

{team}

IMPORTANT: You are coordinating a team working in parallel with other teams on the same task. A shared memory system tracks what other teams have discovered. While carrying out your plan, consider instructing your Assistant agent to check shared memory to see what approaches other teams are taking. This will help you create a more effective plan that avoids duplication and learns from others.

Based on the team composition, and known and unknown facts, please devise a short bullet-point plan for addressing the original request. Remember, there is no requirement to involve all team members -- a team member's particular expertise may not be needed for this task."""


ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT = """
We are working to address the following user request:

{task}


To answer this request we have assembled the following team:

{team}


Here is an initial fact sheet to consider:

{facts}


Here is the plan to follow as best as possible:

{plan}
"""


ORCHESTRATOR_PROGRESS_LEDGER_PROMPT = """
Recall we are working on the following request:

{task}

And we have assembled the following team:

{team}

IMPORTANT: You are coordinating a team working in parallel with other teams on the same task. A shared memory system tracks what other teams have discovered.

BEFORE deciding who should speak next, you MUST check shared memory to see what other parallel teams have already done. This is especially important when:
1. Deciding who should speak next - Verify if others have already completed similar work
2. Evaluating progress - Consider if other teams have found solutions you can learn from
3. When stuck or making slow progress - Check what has worked for other teams

HOW to check: Direct your Assistant agent to run Python code that queries shared memory. The Assistant knows how to execute the queries - instruct them to check shared memory before starting new subtasks.

HOW to use the information:
- If others have completed similar work: Skip redundant steps, learn from their approach, or take a different path
- If others found solutions: Consider adapting their successful methods
- If others failed: Avoid repeating their failed approaches
- If approaches differ: Choose the most promising path based on what's been tried

Decision-making process:
1. BEFORE assigning a subtask → Instruct Assistant to check shared memory first
2. Review the shared memory output → Understand what others have done
3. Adapt your instruction → Avoid duplication, learn from others, or take alternative approaches
4. Assign the adapted subtask → Give specific instructions based on shared memory insights

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and / or getting the same responses as before? Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a handful of times.
    - Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success such as the inability to read from a required file)
    - Who should speak next? (select from: {names})
    - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need. REMEMBER: If assigning a subtask, instruct them to check shared memory first.)

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
       "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "next_speaker": {{
            "reason": string,
            "answer": string (select from: {names})
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
"""


class LedgerEntryBooleanAnswer(BaseModel):
    reason: str
    answer: bool


class LedgerEntryStringAnswer(BaseModel):
    reason: str
    answer: str


class LedgerEntry(BaseModel):
    is_request_satisfied: LedgerEntryBooleanAnswer
    is_in_loop: LedgerEntryBooleanAnswer
    is_progress_being_made: LedgerEntryBooleanAnswer
    next_speaker: LedgerEntryStringAnswer
    instruction_or_question: LedgerEntryStringAnswer


ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT = """As a reminder, we are working to solve the following task:

{task}

It's clear we aren't making as much progress as we would like, but we may have learned something new. Please rewrite the following fact sheet, updating it to include anything new we have learned that may be helpful. Example edits can include (but are not limited to) adding new guesses, moving educated guesses to verified facts if appropriate, etc. Updates may be made to any section of the fact sheet, and more than one section of the fact sheet can be edited. This is an especially good time to update educated guesses, so please at least add or update one educated guess or hunch, and explain your reasoning.

Here is the old fact sheet:

{facts}
"""


ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT = """Please briefly explain what went wrong on this last run (the root cause of the failure), and then come up with a new plan that takes steps and/or includes hints to overcome prior challenges and especially avoids repeating the same mistakes. 

IMPORTANT: Before creating the new plan, instruct your Assistant agent to check shared memory to see what other parallel teams have discovered. Use this information to adapt your plan - learn from what has worked for others and avoid repeating failed approaches.

As before, the new plan should be concise, be expressed in bullet-point form, and consider the following team composition (do not involve any other outside people since we cannot contact anyone else):

{team}
"""


ORCHESTRATOR_FINAL_ANSWER_PROMPT = """
We are working on the following task:
{task}

We have completed the task.

The above messages contain the conversation that took place to complete the task.

Based on the information gathered, provide the final answer to the original request.
The answer should be phrased as if you were speaking to the user.
"""
