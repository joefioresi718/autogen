import asyncio
import os
import re
import logging
import yaml
import warnings
import contextvars
import builtins
import shutil
import json
from datetime import datetime
from typing import List, Optional, Dict
from collections import deque
from autogen_agentchat import TRACE_LOGGER_NAME as AGENTCHAT_TRACE_LOGGER_NAME, EVENT_LOGGER_NAME as AGENTCHAT_EVENT_LOGGER_NAME
from autogen_agentchat.base import TaskResult
from autogen_core import TRACE_LOGGER_NAME as CORE_TRACE_LOGGER_NAME, EVENT_LOGGER_NAME as CORE_EVENT_LOGGER_NAME, CancellationToken
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticMemoryGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
)
from autogen_core.logging import LLMCallEvent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.code_executor import FunctionWithRequirements, CodeBlock, CodeExecutor
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import (
    TextMessage,
    AgentEvent,
    ChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._model_info import _MODEL_TOKEN_LIMITS, resolve_model
from autogen_agentchat.utils import content_to_str

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

core_event_logger = logging.getLogger(CORE_EVENT_LOGGER_NAME)
agentchat_event_logger = logging.getLogger(AGENTCHAT_EVENT_LOGGER_NAME)
agentchat_trace_logger = logging.getLogger(AGENTCHAT_TRACE_LOGGER_NAME)

# Module-level functions that read from a JSON file
# This works across subprocess boundaries since files are shared
# These functions will be serialized and available in executor subprocesses
def get_shared_memory_keys():
    """Get all keys from shared memory. Returns a list of strings.
    
    This function can be called from Python code executed in the executor.
    Example: keys = get_shared_memory_keys()
    
    Note: The shared memory is stored in a JSON file (.shared_memory.json)
    The path is set via SHARED_MEMORY_FILE environment variable.
    """
    import json
    import os
    
    # First try environment variable (set by main process)
    memory_file = os.environ.get('SHARED_MEMORY_FILE')
    if memory_file and os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            data = json.load(f)
            return list(data.keys   ())
    return []

def get_shared_memory_value(key: str):
    """Get a value from shared memory by key. Returns the value or None.
    
    Args:
        key: The key to look up (e.g., "team0-step1-MagenticMemoryOrchestrator")
    
    Returns:
        The value associated with the key, or None if not found.
    
    Example: value = get_shared_memory_value("team0-step1-MagenticMemoryOrchestrator")
    
    Note: The shared memory is stored in a JSON file (.shared_memory.json)
    The path is set via SHARED_MEMORY_FILE environment variable.
    """
    import json
    import os
    
    # First try environment variable (set by main process)
    memory_file = os.environ.get('SHARED_MEMORY_FILE')
    if memory_file and os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            data = json.load(f)
            return data.get(key)
    
    return None

# Create a context variable to hold the current team's log file and the current team id.
current_log_file = contextvars.ContextVar("current_log_file", default=None)
current_team_id = contextvars.ContextVar("current_team_id", default=None)
current_shared_memory = contextvars.ContextVar("current_shared_memory", default=None)

# Custom executor wrapper that automatically imports functions for Python code
class AutoImportCodeExecutor(CodeExecutor):
    """Wrapper around LocalCommandLineCodeExecutor that automatically prepends function imports to Python code."""
    
    def __init__(self, base_executor: LocalCommandLineCodeExecutor):
        self._base_executor = base_executor
        self._functions_module = base_executor.functions_module if hasattr(base_executor, 'functions_module') else 'functions'
        # Check if functions are available by checking the private _functions attribute
        functions_list = getattr(base_executor, '_functions', [])
        self._has_functions = len(functions_list) > 0
    
    async def execute_code_blocks(
        self, 
        code_blocks: List[CodeBlock], 
        cancellation_token: Optional[CancellationToken] = None
    ):
        # Prepend import statement to Python code blocks if functions are available
        modified_blocks = []
        for block in code_blocks:
            if block.language and block.language.lower() == "python":
                # Check if functions are available and if import is not already present
                has_import = "from functions import" in block.code or "import functions" in block.code
                if self._has_functions and not has_import:
                    import_stmt = f"from {self._functions_module} import *\n"
                    modified_code = import_stmt + block.code
                    modified_blocks.append(CodeBlock(code=modified_code, language=block.language))
                else:
                    modified_blocks.append(block)
            else:
                modified_blocks.append(block)
        
        return await self._base_executor.execute_code_blocks(modified_blocks, cancellation_token)
    
    async def start(self):
        return await self._base_executor.start()
    
    async def stop(self):
        return await self._base_executor.stop()
    
    async def restart(self):
        return await self._base_executor.restart()
    
    @property
    def work_dir(self):
        return self._base_executor.work_dir

# Shared memory class for inter-team communication
class SharedMemory:
    """Thread-safe shared memory for parallel teams to store and retrieve outputs."""
    
    def __init__(self, persistence_file: str = ".shared_memory.json", summarizer_client: Optional[ChatCompletionClient] = None):
        self._memory: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._step_counters: Dict[int, int] = {}  # Track step numbers per team
        self._persistence_file = persistence_file
        self._summarizer_client = summarizer_client
    
    async def _persist_to_file(self):
        """Persist memory to JSON file for cross-process access."""
        try:
            import json
            with open(self._persistence_file, 'w') as f:
                json.dump(self._memory, f)
        except Exception as e:
            # Don't fail if file write fails
            print(f"Warning: Failed to persist shared memory to file: {e}", flush=True)
    
    async def _summarize_step(self, source: str, content: str) -> str:
        messages = [
            UserMessage(
                content=(
                    "Summarize the agent step purpose in 15–20 words as a concise phrase, no quotes or trailing punctuation.\n\n"
                    f"Source: {source}\n\nOutput:\n{content[:2000]}"
                ),
                source="SharedMemory"
            )
        ]
        response = await self._summarizer_client.create(messages)
        assert isinstance(response.content, str)
        return response.content.strip().replace("\n", " ")
    
    async def store(self, team_idx: int, source: str, content: str) -> str:
        """Store a message in shared memory and return the key."""
        async with self._lock:
            # Initialize step counter for team if needed
            if team_idx not in self._step_counters:
                self._step_counters[team_idx] = 0
            
            # Increment step counter for this team
            self._step_counters[team_idx] += 1
            step_num = self._step_counters[team_idx]
            
            # Create key with concise summary
            summary = await self._summarize_step(source, content)
            key = f"team{team_idx}-step{step_num} - {summary}"
            self._memory[key] = content
            
            # Persist to file for cross-process access
            await self._persist_to_file()
            
            return key
    
    async def get(self, key: str) -> Optional[str]:
        """Retrieve a value from shared memory by key."""
        async with self._lock:
            return self._memory.get(key)
    
    async def get_all(self) -> Dict[str, str]:
        """Get a copy of all shared memory."""
        async with self._lock:
            return self._memory.copy()
    
    async def get_keys(self) -> List[str]:
        """Get a list of all keys in shared memory."""
        async with self._lock:
            return list(self._memory.keys())
    
    def __len__(self) -> int:
        """Return the number of entries in shared memory (not thread-safe, for debugging)."""
        return len(self._memory)

# Save the original print function and event_logger.info method.
original_print = builtins.print
original_agentchat_event_logger_info = agentchat_event_logger.info
original_core_event_logger_info = core_event_logger.info

class LogHandler(logging.FileHandler):
    def __init__(self, filename: str = "log.jsonl", print_message: bool = True) -> None:
        super().__init__(filename, mode="w")
        self.print_message = print_message

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).isoformat()
            if AGENTCHAT_EVENT_LOGGER_NAME in record.name:
                original_msg = record.msg
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": content_to_str(record.msg.content),
                        "type": record.msg.type,
                    }
                )
                super().emit(record)
                record.msg = original_msg
            elif CORE_EVENT_LOGGER_NAME in record.name:
                if isinstance(record.msg, LLMCallEvent):
                    original_msg = record.msg
                    record.msg = json.dumps(
                        {
                            "timestamp": ts,
                            "prompt_tokens": record.msg.kwargs["prompt_tokens"],
                            "completion_tokens": record.msg.kwargs["completion_tokens"],
                            "type": "LLMCallEvent",
                        }
                    )
                    super().emit(record)
                    record.msg = original_msg
        except Exception:
            print("error in logHandler.emit", flush=True)
            self.handleError(record)

def tee_print(*args, **kwargs):
    # Get the current log file from the context.
    log_file = current_log_file.get()
    # Call the original print (goes to the console).
    original_print(*args, **kwargs)
    # Also write to the log file if one is set.
    if log_file is not None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(map(str, args)) + end
        log_file.write(message)
        log_file.flush()

def team_specific_agentchat_event_logger_info(msg, *args, **kwargs):
    team_id = current_team_id.get()
    if team_id is not None:
        # Get a logger with a team-specific name.
        team_logger = logging.getLogger(f"{AGENTCHAT_EVENT_LOGGER_NAME}.team{team_id}")
        team_logger.info(msg, *args, **kwargs)
    else:
        original_agentchat_event_logger_info(msg, *args, **kwargs)

def team_specific_core_event_logger_info(msg, *args, **kwargs):
    team_id = current_team_id.get()
    if team_id is not None:
        # Get a logger with a team-specific name.
        team_logger = logging.getLogger(f"{CORE_EVENT_LOGGER_NAME}.team{team_id}")
        team_logger.info(msg, *args, **kwargs)
    else:
        original_core_event_logger_info(msg, *args, **kwargs)

# Monkey-patch the built-in print and event_logger.info methods with our team-specific versions.
builtins.print = tee_print
agentchat_event_logger.info = team_specific_agentchat_event_logger_info
core_event_logger.info = team_specific_core_event_logger_info

async def intercept_messages_for_shared_memory(
    stream,
    team_idx: int,
    shared_memory: SharedMemory
):
    """Intercept messages from a stream and store them in shared memory."""
    async for message in stream:
        # Skip TaskResult - it's the final result, not a message to store
        # Also skip StopMessage as it's just a signal
        if isinstance(message, (TaskResult, StopMessage)):
            yield message
            continue
        
        # Extract source and content from message
        source = getattr(message, 'source', 'unknown')
        
        # Convert message content to string
        if hasattr(message, 'to_model_text'):
            content = message.to_model_text()
        elif hasattr(message, 'content'):
            content = content_to_str(message.content)
        else:
            content = str(message)
        
        # Store in shared memory (non-blocking, store in background)
        try:
            await shared_memory.store(team_idx, source, content)
        except Exception as e:
            # Log but don't fail on shared memory errors
            print(f"Warning: Failed to store message in shared memory: {e}", flush=True)
        
        # Yield the message to continue the stream
        yield message

async def run_team(
    team: MagenticMemoryGroupChat, 
    team_idx: int, 
    task: str, 
    cancellation_token: CancellationToken, 
    logfile,
    shared_memory: Optional[SharedMemory] = None
):
    token_logfile = current_log_file.set(logfile)
    token_team_id = current_team_id.set(team_idx)
    token_shared_memory = None
    if shared_memory is not None:
        token_shared_memory = current_shared_memory.set(shared_memory)
    try:
        # Get the raw stream
        stream = team.run_stream(
            task=task.strip(),
            cancellation_token=cancellation_token
        )
        
        # Intercept messages if shared memory is provided
        if shared_memory is not None:
            stream = intercept_messages_for_shared_memory(stream, team_idx, shared_memory)
        
        # Pass to Console for display
        task_result = await Console(stream)
        return team_idx, task_result
    finally:
        current_log_file.reset(token_logfile)
        current_team_id.reset(token_team_id)
        if token_shared_memory is not None:
            current_shared_memory.reset(token_shared_memory)
        logfile.close()

async def aggregate_final_answer(task: str, client: ChatCompletionClient, team_results, source: str = "Aggregator", cancellation_token: Optional[CancellationToken] = None) -> str:
        """
        team_results: {"team_key": TaskResult}
        team_completion_order: The order in which the teams completed their tasks
        """

        if len(team_results) == 1:
            final_answer = list(team_results.values())[0].messages[-1].content
            aggregator_logger.info(
                f"{source} (Response):\n{final_answer}"
            )
            return final_answer

        assert len(team_results) > 1

        aggregator_messages_to_send = {team_id: deque() for team_id in team_results.keys()} # {team_id: context}

        team_ids = list(team_results.keys())
        current_round = 0
        while (
            not all(len(team_result.messages) == 0 for team_result in team_results.values())
            and ((not resolve_model(client._create_args["model"]) in _MODEL_TOKEN_LIMITS) or client.remaining_tokens([m for messages in aggregator_messages_to_send.values() for m in messages])
            > 2000)
        ):
            team_idx = team_ids[current_round % len(team_ids)]
            if len(team_results[team_idx].messages) > 0:
                m = team_results[team_idx].messages[-1]
                if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                    # Ignore tool call messages.
                    pass
                elif isinstance(m, StopMessage | HandoffMessage):
                    aggregator_messages_to_send[team_idx].appendleft(UserMessage(content=m.to_model_text(), source=m.source))
                elif m.source == "MagenticMemoryOrchestrator":
                    assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                    aggregator_messages_to_send[team_idx].appendleft(AssistantMessage(content=m.to_model_text(), source=m.source))
                else:
                    assert isinstance(m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage))
                    aggregator_messages_to_send[team_idx].appendleft(UserMessage(content=m.to_model_text(), source=m.source))
                team_results[team_idx].messages.pop()
            current_round += 1

        # Log the messages to send
        payload = ""
        for team_idx, messages in aggregator_messages_to_send.items():
            payload += f"\n{'*'*75} \n" f"Team #: {team_idx}" f"\n{'*'*75} \n"
            for message in messages:
                payload += f"\n{'-'*75} \n" f"{message.source}:\n" f"\n{message.content}\n"
            payload += f"\n{'-'*75} \n" f"Team #{team_idx} stop reason:\n" f"\n{team_results[team_idx].stop_reason}\n"
        payload += f"\n{'*'*75} \n"
        aggregator_logger.info(f"{source} (Aggregator Messages):\n{payload}")

        context: List[LLMMessage] = []

        # Add the preamble
        context.append(
            UserMessage(
                content=f"Earlier you were asked the following:\n\n{task}\n\nYour team then worked diligently to address that request. You have been provided with a collection of transcripts and stop reasons from {len(team_results)} different teams to the question. Your task is to carefully evaluate the correctness of each team's response by analyzing their respective transcripts and stop reasons. After considering all perspectives, provide a FINAL ANSWER to the question. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.",
                source=source,
            )
        )

        for team_idx, aggregator_messages in aggregator_messages_to_send.items():
            context.append(
                UserMessage(
                    content=f"Transcript from Team #{team_idx}:",
                    source=source,
                )
            )
            for message in aggregator_messages:
                context.append(message)
            context.append(
                UserMessage(
                    content=f"Stop reason from Team #{team_idx}:",
                    source=source,
                )
            )
            context.append(
                UserMessage(
                    content=team_results[team_idx].stop_reason if team_results[team_idx].stop_reason else "No stop reason provided.",
                    source=source,
                )
            )

        # ask for the final answer
        context.append(
            UserMessage(
                content=f"""
    Let's think step-by-step. Carefully review the conversation above, critically evaluate the correctness of each team's response, and then output a FINAL ANSWER to the question. The question is repeated here for convenience:

    {task}

    To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
    Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
    ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
    If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
    """.strip(),
                source=source,
            )
        )

        response = await client.create(context, cancellation_token=cancellation_token)
        assert isinstance(response.content, str)

        final_answer = re.sub(r"FINAL ANSWER:", "[FINAL ANSWER]:", response.content)
        aggregator_logger.info(
            f"{source} (Response):\n{final_answer}"
        )

        return re.sub(r"FINAL ANSWER:", "FINAL AGGREGATED ANSWER:", response.content)


async def main(num_teams: int, num_answers: int) -> None:

    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(config["web_surfer_client"])
    file_surfer_client = ChatCompletionClient.load_component(config["file_surfer_client"])

    # Read the prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read().strip()
    filename = "__FILE_NAME__".strip()

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    
    # Orchestrator-specific instructions for shared memory
    orchestrator_instruction = """
[IMPORTANT INSTRUCTIONS FOR ORCHESTRATOR]

You are coordinating a team working in parallel with other teams on the same task. A shared memory system tracks what other teams have discovered.

WHEN to check shared memory (mandatory checkpoints):
1. BEFORE creating your initial plan - Check what approaches other teams are taking
2. BEFORE deciding who should speak next - Verify if others have already completed similar work
3. WHEN evaluating progress - Consider if other teams have found solutions you can learn from
4. BEFORE updating your plan - Adapt based on what has worked or failed for other teams
5. WHEN stuck or making slow progress - Check what has worked for other teams

HOW to check: Direct your Assistant agent to run Python code that queries shared memory. The Assistant knows how to execute the queries - you just need to instruct them to do it before starting new subtasks.

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

Key principle: Shared memory checking is REQUIRED before every major decision. Use it to coordinate, avoid waste, and learn from parallel teams.

[END OF ORCHESTRATOR INSTRUCTIONS]
"""
    
    task = f"{prompt}\n\n{filename_prompt}"

    # Reset logs directory (remove all files in it)
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    # Create shared memory for inter-team communication
    # Use absolute path for JSON file so it's accessible from executor subprocesses
    # Store it in the current working directory where the script runs
    memory_file_path = os.path.abspath('.shared_memory.json')
    shared_memory = SharedMemory(persistence_file=memory_file_path, summarizer_client=orchestrator_client)
    
    # Set environment variable so executor subprocesses can find the file
    os.environ['SHARED_MEMORY_FILE'] = memory_file_path
    
    # Initialize the JSON file so it exists when executor tries to read it
    import json
    with open(memory_file_path, 'w') as f:
        json.dump({}, f)

    # Shared memory instructions that will be included in agent descriptions
    shared_memory_info = """
    
You have access to shared memory functions to query what other parallel teams have discovered. These functions are automatically available in Python code (no imports needed):

- get_shared_memory_keys() → Returns list of all keys (e.g., ["team0-step1 - Extracted table from PDF", "team1-step2 - Verified answer via web search"])
- get_shared_memory_value(key) → Returns value for a key, or None if not found

Basic usage pattern:
```python
keys = get_shared_memory_keys()
print(f"Found {len(keys)} entries from other teams")
if keys:
    # Check recent entries
    for key in keys[-5:]:
        value = get_shared_memory_value(key)
        print(f"{key}: {str(value)[:200]}...")
print("=== End shared memory check ===")
```

When instructed to check shared memory, execute the query code and return the results clearly formatted.
"""
    
    teams = []
    async_tasks = []
    tokens = []
    team_directories_to_cleanup = []  # Track directories for cleanup
    for team_idx in range(num_teams):
        # Set up the team
        # Add shared memory instructions to Assistant's description so it appears in team description
        assistant_description = f"A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills.{shared_memory_info}"
        coder = MagenticOneCoderAgent(
            "Assistant",
            model_client = coder_client,
        )
        # Override the description to include shared memory info
        coder._description = assistant_description

        # Create executor with shared memory helper functions
        # These functions will be available when Python code is executed
        # They read from the JSON file that SharedMemory persists to
        # Wrap in AutoImportCodeExecutor to automatically import functions
        base_executor = LocalCommandLineCodeExecutor(
            functions=[get_shared_memory_keys, get_shared_memory_value]
        )
        executor = CodeExecutorAgent(
            "ComputerTerminal", 
            code_executor=AutoImportCodeExecutor(base_executor)
        )

        file_surfer = FileSurfer(
            name="FileSurfer",
            model_client = file_surfer_client,
        )

        # Create team-specific downloads folder
        team_downloads_folder = f"downloads_team_{team_idx}"
        os.makedirs(team_downloads_folder, exist_ok=True)
        team_debug_dir = os.path.join(logs_dir, f"team_{team_idx}")
        os.makedirs(team_debug_dir, exist_ok=True)
        # Track directories for cleanup
        team_directories_to_cleanup.append(team_downloads_folder)
        team_directories_to_cleanup.append(team_debug_dir)

        web_surfer = MultimodalWebSurfer(
            name="WebSurfer",
            model_client = web_surfer_client,
            downloads_folder=team_downloads_folder,
            debug_dir=team_debug_dir,
            to_save_screenshots=True,
        )
        team = MagenticMemoryGroupChat(
            [coder, executor, file_surfer, web_surfer],
            model_client=orchestrator_client,
            max_turns=30,
            final_answer_prompt= f""",
We have completed the following task:

{prompt}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip()
        )
        teams.append(team)
        cancellation_token = CancellationToken()
        tokens.append(cancellation_token)
        logfile = open(f"console_log_{team_idx}.txt", "w")
        team_agentchat_logger = logging.getLogger(f"{AGENTCHAT_EVENT_LOGGER_NAME}.team{team_idx}")
        team_core_logger = logging.getLogger(f"{CORE_EVENT_LOGGER_NAME}.team{team_idx}")
        team_log_handler = LogHandler(f"log_{team_idx}.jsonl", print_message=False)
        team_agentchat_logger.addHandler(team_log_handler)
        team_core_logger.addHandler(team_log_handler)
        async_task = asyncio.create_task(
            run_team(team, team_idx, task, cancellation_token, logfile, shared_memory)
        )
        async_tasks.append(async_task)

    # Wait until at least num_answers tasks have completed.
    team_results = {}
    for future in asyncio.as_completed(async_tasks):
        try:
            team_id, result = await future
            team_results[team_id] = result
        except Exception as e:
            # Optionally log exception.
            print(f"Task raised an exception: {e}")
        if len(team_results) >= num_answers:
            break

    # Cancel any pending teams.
    for task, token in zip(async_tasks, tokens):
        if not task.done():
            token.cancel()
    # Await all tasks to handle cancellation gracefully.
    await asyncio.gather(*async_tasks, return_exceptions=True)

    print("len(team_results):", len(team_results))
    final_answer = await aggregate_final_answer(prompt, orchestrator_client, team_results)
    print(final_answer)

    # Print shared memory summary (for debugging/inspection)
    all_memory = await shared_memory.get_all()
    print(f"\nShared Memory Summary: {len(all_memory)} entries")
    for key in sorted(all_memory.keys()):
        content_preview = all_memory[key][:100] + "..." if len(all_memory[key]) > 100 else all_memory[key]
        print(f"  {key}: {content_preview}")

    # Cleanup team-specific directories
    for directory in team_directories_to_cleanup:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except Exception as e:
                print(f"Warning: Failed to clean up directory {directory}: {e}")
    
    # Cleanup shared memory JSON file
    if os.path.exists(memory_file_path):
        try:
            os.remove(memory_file_path)
        except Exception as e:
            print(f"Warning: Failed to clean up shared memory file {memory_file_path}: {e}")

if __name__ == "__main__":
    num_teams = 3
    num_answers = 3

    agentchat_trace_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("trace.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    agentchat_trace_logger.addHandler(file_handler)

    core_event_logger.setLevel(logging.DEBUG)
    agentchat_event_logger.setLevel(logging.DEBUG)
    log_handler = LogHandler()
    core_event_logger.addHandler(log_handler)
    agentchat_event_logger.addHandler(log_handler)

    # Create another logger for the aggregator
    aggregator_logger = logging.getLogger("aggregator")
    aggregator_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("aggregator_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    aggregator_logger.addHandler(fh)


    asyncio.run(main(num_teams, num_answers))
