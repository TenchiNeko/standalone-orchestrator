"""
Standalone Agent Runner — Direct LLM API + Tool Execution Loop.

This is the core of the standalone system. It:
1. Calls Ollama (or Anthropic) HTTP API directly — no CLI wrappers
2. Defines tools (run_command, write_file, read_file, list_directory)
3. Implements the agentic loop: send prompt → get response → if tool_call,
   execute it locally → feed result back → repeat until text response
4. Zero dependency on external CLI agent wrappers

The tool execution loop is what makes agents actually DO things instead of
just generating text about what they would do.
"""

import os
import json
import time
import subprocess
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests

from standalone_config import Config, AgentConfig, ModelConfig
from standalone_models import TaskState, AgentResult, DoD, DoDCriterion

logger = logging.getLogger(__name__)


# ============================================================
# Tool Definitions (sent to the LLM so it knows what it can call)
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the working directory. Returns stdout, stderr, and exit code. Use for: running tests, git commands, installing packages, compiling, checking file existence, grep, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (run via bash -c)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60)",
                        "default": 60
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed. Use for creating new files or overwriting existing ones completely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content as a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path. Returns a directory listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to working directory (default: '.')",
                        "default": "."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list recursively (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific string in a file with new content. The old_str must appear exactly once in the file. Use for surgical edits to existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to working directory"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to find (must be unique in the file)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    }
]


# ============================================================
# Tool Executor — actually runs the tools on the local system
# ============================================================

class ToolExecutor:
    """Executes tool calls from the LLM on the local filesystem."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "run_command":
                return self._run_command(arguments)
            elif tool_name == "write_file":
                return self._write_file(arguments)
            elif tool_name == "read_file":
                return self._read_file(arguments)
            elif tool_name == "list_directory":
                return self._list_directory(arguments)
            elif tool_name == "edit_file":
                return self._edit_file(arguments)
            else:
                return f"ERROR: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _run_command(self, args: dict) -> str:
        command = args.get("command", "")
        timeout = args.get("timeout", 60)

        if not command:
            return "ERROR: No command provided"

        # Safety rails: block commands that would destroy orchestrator files
        PROTECTED_PATTERNS = [
            "rm -rf .",
            "rm -rf /",
            "rm -rf *",
            "git rm -rf .",
            "git rm -rf *",
            "git rm -rf .agents",
            "git rm -rf prompts",
            "git clean -fdx",
            "sudo ",  # No sudo access
        ]
        # Block deletion of orchestrator files
        PROTECTED_FILES = [
            "standalone_main.py", "standalone_agents.py", "standalone_config.py",
            "standalone_models.py", "standalone_orchestrator.py", "standalone_session.py",
            "test_standalone_integration.py",
        ]
        PROTECTED_DIRS = ["prompts", ".agents"]
        cmd_lower = command.lower().strip()
        for pattern in PROTECTED_PATTERNS:
            if pattern in cmd_lower:
                return f"ERROR: Blocked dangerous command: '{command[:80]}'. This command could damage the orchestrator."
        if "rm " in cmd_lower or "git rm" in cmd_lower:
            for pf in PROTECTED_FILES:
                if pf in command:
                    return f"ERROR: Cannot delete protected orchestrator file: {pf}"
            for pd in PROTECTED_DIRS:
                if pd in command and ("rm " in cmd_lower or "git rm" in cmd_lower):
                    return f"ERROR: Cannot delete protected directory: {pd}"

        logger.debug(f"  TOOL run_command: {command[:100]}")

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += ("\n" if output else "") + f"STDERR: {result.stderr}"
            output += f"\nEXIT_CODE: {result.returncode}"

            # Truncate very long output
            if len(output) > 10000:
                output = output[:5000] + "\n\n... (truncated) ...\n\n" + output[-3000:]

            return output

        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"

    def _write_file(self, args: dict) -> str:
        path = args.get("path", "")
        content = args.get("content", "")

        if not path:
            return "ERROR: No path provided"

        full_path = self.working_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

        logger.debug(f"  TOOL write_file: {path} ({len(content)} bytes)")
        return f"OK: Wrote {len(content)} bytes to {path}"

    def _read_file(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return "ERROR: No path provided"

        full_path = self.working_dir / path
        if not full_path.exists():
            return f"ERROR: File not found: {path}"
        if not full_path.is_file():
            return f"ERROR: Not a file: {path}"

        content = full_path.read_text()

        # Truncate very large files
        if len(content) > 15000:
            content = content[:7000] + "\n\n... (truncated) ...\n\n" + content[-5000:]

        logger.debug(f"  TOOL read_file: {path} ({len(content)} bytes)")
        return content

    def _list_directory(self, args: dict) -> str:
        path = args.get("path", ".")
        recursive = args.get("recursive", False)

        full_path = self.working_dir / path
        if not full_path.exists():
            return f"ERROR: Directory not found: {path}"

        lines = []
        if recursive:
            for p in sorted(full_path.rglob("*")):
                if any(part.startswith(".") for part in p.relative_to(full_path).parts):
                    if not str(p.relative_to(full_path)).startswith(".agents"):
                        continue
                rel = p.relative_to(full_path)
                prefix = "📁 " if p.is_dir() else "📄 "
                lines.append(f"{prefix}{rel}")
        else:
            for p in sorted(full_path.iterdir()):
                prefix = "📁 " if p.is_dir() else "📄 "
                lines.append(f"{prefix}{p.name}")

        if len(lines) > 200:
            lines = lines[:200] + [f"... and {len(lines) - 200} more"]

        return "\n".join(lines) if lines else "(empty directory)"

    def _edit_file(self, args: dict) -> str:
        path = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")

        if not path:
            return "ERROR: No path provided"

        full_path = self.working_dir / path
        if not full_path.exists():
            return f"ERROR: File not found: {path}"

        content = full_path.read_text()
        count = content.count(old_str)

        if count == 0:
            return f"ERROR: old_str not found in {path}"
        if count > 1:
            return f"ERROR: old_str appears {count} times in {path} (must be unique)"

        new_content = content.replace(old_str, new_str, 1)
        full_path.write_text(new_content)

        logger.debug(f"  TOOL edit_file: {path}")
        return f"OK: Replaced text in {path}"


# ============================================================
# LLM Client — talks to Ollama or Anthropic HTTP API directly
# ============================================================

class LLMClient:
    """
    Direct HTTP client for LLM APIs.

    Supports:
    - Ollama: POST /api/chat with tool support
    - Anthropic: POST /v1/messages (fallback for bootstrapping)
    """

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return the raw response.

        Returns dict with:
          - "content": str (text response)
          - "tool_calls": list of tool call objects (if any)
          - "done": bool
        """
        if self.config.provider == "ollama":
            return self._chat_ollama(messages, tools, temperature)
        elif self.config.provider == "anthropic":
            return self._chat_anthropic(messages, tools, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _chat_ollama(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call Ollama /api/chat endpoint."""
        endpoint = (self.config.endpoint or "http://127.0.0.1:11434").rstrip("/")
        url = f"{endpoint}/api/chat"

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }

        if tools and self.config.supports_tools:
            payload["tools"] = tools

        try:
            resp = self.session.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {endpoint}: {e}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out after 600s")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

        # Parse Ollama response format
        message = data.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        return {
            "content": content,
            "tool_calls": tool_calls,
            "done": data.get("done", True),
            "total_duration": data.get("total_duration"),
            "eval_count": data.get("eval_count"),
        }

    def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call Anthropic /v1/messages endpoint."""
        api_key = os.environ.get(self.config.api_key_env or "ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(f"API key not found in env var: {self.config.api_key_env}")

        url = "https://api.anthropic.com/v1/messages"

        # Convert messages to Anthropic format
        # Anthropic uses system as a top-level param, not in messages
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                api_messages.append(msg)

        payload = {
            "model": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "messages": api_messages,
        }
        if system_text:
            payload["system"] = system_text.strip()
        if temperature is not None:
            payload["temperature"] = temperature

        # Convert tools to Anthropic format
        if tools:
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function":
                    func = t["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func["description"],
                        "input_schema": func["parameters"]
                    })
            if anthropic_tools:
                payload["tools"] = anthropic_tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

        # Parse Anthropic response
        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "function": {
                        "name": block["name"],
                        "arguments": block["input"],  # already a dict
                    },
                    "id": block["id"],
                })

        return {
            "content": content_text,
            "tool_calls": tool_calls,
            "done": data.get("stop_reason") != "tool_use",
        }


# ============================================================
# AgentRunner — the agentic tool-use loop
# ============================================================

class AgentRunner:
    """
    Runs agents with real tool execution.

    The core loop:
    1. Send prompt + tools to LLM
    2. If response contains tool_calls → execute each tool locally
    3. Append tool results to conversation
    4. Send back to LLM for next step
    5. Repeat until LLM responds with just text (no more tool calls)
    """

    def __init__(self, config: Config, working_dir: Path):
        self.config = config
        self.working_dir = working_dir
        self.tool_executor = ToolExecutor(working_dir)
        self.prompts_dir = Path(__file__).parent
        self._llm_clients: Dict[str, LLMClient] = {}

    def _get_llm_client(self, model_config: ModelConfig) -> LLMClient:
        """Get or create an LLM client for the given model config."""
        key = f"{model_config.provider}:{model_config.endpoint}:{model_config.model_id}"
        if key not in self._llm_clients:
            self._llm_clients[key] = LLMClient(model_config)
        return self._llm_clients[key]

    def _load_prompt(self, prompt_file: str) -> str:
        prompt_path = self.prompts_dir / prompt_file
        if prompt_path.exists():
            return prompt_path.read_text()
        # Also check relative to working dir
        alt_path = self.working_dir / prompt_file
        if alt_path.exists():
            return alt_path.read_text()
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""

    def _run_agent(
        self,
        agent_config: AgentConfig,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[dict]] = None,
    ) -> AgentResult:
        """
        THE CORE AGENTIC LOOP.

        Sends the prompt to the LLM with tool definitions, then loops:
        - If the LLM returns tool calls → execute them → feed results back
        - If the LLM returns text only → done, return the result
        """
        client = self._get_llm_client(agent_config.model)
        max_rounds = agent_config.max_tool_rounds
        start_time = time.time()

        # Build initial messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        all_output = []  # Collect all text output across rounds
        round_count = 0

        logger.info(f"Running {agent_config.role} agent...")
        logger.debug(f"  Model: {agent_config.model.name} ({agent_config.model.model_id})")
        logger.debug(f"  Endpoint: {agent_config.model.endpoint}")

        try:
            while round_count < max_rounds:
                round_count += 1

                # Call the LLM
                response = client.chat(messages, tools=tools)

                content = response.get("content", "")
                tool_calls = response.get("tool_calls", [])

                # FALLBACK: If model embeds tool calls as JSON in text content
                # (common with Qwen via Ollama — outputs tool JSON in content
                #  instead of structured tool_calls)
                if not tool_calls and content and tools:
                    parsed_calls = self._extract_tool_calls_from_text(content)
                    if parsed_calls:
                        tool_calls = parsed_calls
                        # Strip the tool-call JSON from the text output
                        content = self._strip_tool_json_from_text(content)
                        logger.debug(f"  Extracted {len(tool_calls)} tool call(s) from text content")

                if content:
                    all_output.append(content)

                # If no tool calls, we're done
                if not tool_calls:
                    logger.debug(f"  Agent finished after {round_count} rounds")
                    break

                # Process tool calls
                logger.debug(f"  Round {round_count}: {len(tool_calls)} tool call(s)")

                # Add assistant message with tool calls to conversation
                assistant_msg = {"role": "assistant", "content": content or ""}
                if agent_config.model.provider == "ollama":
                    assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)

                # Execute each tool call and add results
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args = func.get("arguments", {})

                    # Ollama sometimes returns arguments as a string
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {"command": tool_args}

                    # Execute the tool
                    tool_result = self.tool_executor.execute(tool_name, tool_args)

                    logger.debug(f"  Tool {tool_name}: {tool_result[:100]}...")

                    # Add tool result to conversation
                    # Format depends on provider
                    if agent_config.model.provider == "anthropic":
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tc.get("id", ""),
                                "content": tool_result
                            }]
                        })
                    else:
                        # Ollama format
                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                        })

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > agent_config.timeout_seconds:
                    logger.warning(f"  Agent timed out after {elapsed:.0f}s")
                    return AgentResult(
                        success=False,
                        output="\n".join(all_output),
                        error=f"Timeout after {elapsed:.0f}s ({round_count} rounds)",
                        duration_seconds=elapsed
                    )

            duration = time.time() - start_time

            if round_count >= max_rounds:
                logger.warning(f"  Agent hit max rounds ({max_rounds})")
                return AgentResult(
                    success=False,
                    output="\n".join(all_output),
                    error=f"Hit max tool rounds ({max_rounds})",
                    duration_seconds=duration
                )

            return AgentResult(
                success=True,
                output="\n".join(all_output),
                duration_seconds=duration
            )

        except (ConnectionError, TimeoutError) as e:
            duration = time.time() - start_time
            logger.error(f"  Connection error: {e}")
            return AgentResult(
                success=False, output="\n".join(all_output),
                error=str(e), duration_seconds=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"  Agent execution failed: {e}")
            return AgentResult(
                success=False, output="\n".join(all_output),
                error=str(e), duration_seconds=duration
            )

    # ============================================================
    # Fallback: extract tool calls from text content
    # ============================================================

    def _extract_tool_calls_from_text(self, text: str) -> List[dict]:
        """
        Extract tool calls when the model puts them as JSON in text content
        instead of in the structured tool_calls field.

        Handles formats like:
          {"name": "write_file", "arguments": {"path": "hello.py", "content": "..."}}
        Also handles Qwen artifacts like <|im_start|> tokens.
        """
        tool_calls = []
        known_tools = {"run_command", "write_file", "read_file", "list_directory", "edit_file"}

        # Clean up common model artifacts
        cleaned = text.replace("<|im_start|>", "\n").replace("<|im_end|>", "\n")

        # Strategy: find each {"name": "tool_name" and try to parse the full
        # JSON object starting from that position using json.JSONDecoder
        decoder = json.JSONDecoder()

        # Find all positions where a tool call JSON might start
        for match in re.finditer(r'\{\s*"name"\s*:\s*"(\w+)"', cleaned):
            tool_name = match.group(1)
            if tool_name not in known_tools:
                continue

            start = match.start()
            try:
                obj, end = decoder.raw_decode(cleaned, start)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    tool_calls.append({
                        "function": {
                            "name": obj["name"],
                            "arguments": obj["arguments"]
                        }
                    })
            except (json.JSONDecodeError, ValueError):
                continue

        # Deduplicate — model sometimes repeats the same call
        seen = set()
        unique_calls = []
        for tc in tool_calls:
            key = json.dumps(tc, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_calls.append(tc)

        return unique_calls

    def _strip_tool_json_from_text(self, text: str) -> str:
        """Remove tool-call JSON from text content, keeping any surrounding prose."""
        cleaned = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")

        # Remove JSON tool call blocks by finding and removing them
        decoder = json.JSONDecoder()
        known_tools = {"run_command", "write_file", "read_file", "list_directory", "edit_file"}

        # Find all tool call JSON positions and remove them
        for match in re.finditer(r'\{\s*"name"\s*:\s*"(\w+)"', cleaned):
            if match.group(1) not in known_tools:
                continue
            try:
                _, end = decoder.raw_decode(cleaned, match.start())
                # Mark this region for removal
                cleaned = cleaned[:match.start()] + " " + cleaned[end:]
            except (json.JSONDecodeError, ValueError):
                continue

        # Clean up whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    # ============================================================
    # Role-specific agent methods
    # ============================================================

    def run_initializer(self, goal: str, task_id: str) -> AgentResult:
        agent_config = self.config.get_agent("initializer")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/initializer.txt")
        if not system_prompt:
            system_prompt = "You are the INITIALIZER agent. Set up the project environment for the given task."

        user_prompt = f"""## Task
{goal}

## Task ID
{task_id}

## Instructions
1. Run `pwd` and `ls -la` to understand the current directory
2. Analyze the goal and break it into specific, testable features
3. Create feature_list.json with the feature breakdown
4. Make an initial git commit: `git add -A && git commit -m "chore: initialize task {task_id}"`

Start by exploring the current directory.
"""
        return self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

    def run_explore(self, state: TaskState) -> AgentResult:
        agent_config = self.config.get_agent("explore")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/explore.txt")
        if not system_prompt:
            system_prompt = "You are the EXPLORE agent. Gather context about the codebase. READ files, don't modify them."

        user_prompt = f"""## Task
{state.goal}

## Task ID
{state.task_id}

## Iteration
{state.iteration}

## Instructions
1. Run `ls -la` and examine the project structure
2. Read key files to understand the codebase
3. Identify patterns, dependencies, and relevant code
4. Summarize your findings for the planning agent

Focus on what's relevant to the task. Be selective.
"""
        result = self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)
        if result.success:
            result.exploration_summary = result.output
        return result

    def run_plan(self, state: TaskState) -> AgentResult:
        agent_config = self.config.get_agent("plan")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/plan.txt")
        if not system_prompt:
            system_prompt = "You are the PLAN agent. Create detailed implementation plans with measurable Definition of Done criteria."

        failure_context = ""
        if state.failure_history:
            failure_context = "\n## Previous Failures (MUST ADDRESS)\n"
            for f in state.failure_history[-3:]:
                failure_context += f"- Iteration {f.get('iteration')}: [{f.get('phase')}] {f.get('error')}\n"
                if f.get('rca'):
                    failure_context += f"  RCA: {f.get('rca')}\n"

        exploration_context = ""
        if state.exploration_context:
            exploration_context = f"\n## Exploration Findings\n{state.exploration_context[:2000]}\n"

        user_prompt = f"""## Task
{state.goal}

## Task ID
{state.task_id}

## Iteration
{state.iteration}
{failure_context}
{exploration_context}

## Instructions

Create a detailed implementation plan. You MUST include:

1. **Definition of Done** — specific, testable criteria in this EXACT format:
```dod
- [ ] Criterion 1: specific measurable outcome
- [ ] Criterion 2: specific measurable outcome
```

2. **Step-by-step implementation plan** — what files to create/modify, in what order

3. **Verification commands** — bash commands to verify each criterion

## Rules
- NO CODE in this response — planning only
- Each criterion must be independently testable with a bash command
- Be specific — "works correctly" is not measurable
- If this is a retry, your plan MUST address previous failures
"""
        result = self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

        if result.success:
            result.plan = result.output
            result.dod = self._parse_dod_from_output(result.output)

        return result

    def run_build(self, state: TaskState) -> AgentResult:
        agent_config = self.config.get_agent("build")

        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/build.txt")
        if not system_prompt:
            system_prompt = """You are the BUILD agent. You implement the plan by writing actual code and running commands.

You have tools available: run_command, write_file, read_file, list_directory, edit_file.
USE THEM. Do not just describe what you would do — actually do it using the tools."""

        plan_context = state.current_plan or "No plan available — implement based on the goal."

        user_prompt = f"""## Task
{state.goal}

## Task ID
{state.task_id}

## Iteration
{state.iteration}

## The Plan (FOLLOW THIS)
{plan_context}

## Definition of Done
{state.dod.to_markdown() if state.dod else "No DoD defined."}

## CRITICAL INSTRUCTIONS

You MUST use the tools to actually implement the plan:
- Use `write_file` to create files
- Use `run_command` to execute shell commands
- Use `read_file` to check existing files
- Use `edit_file` to make surgical changes

After implementing, commit your work:
- run_command: "git add -A && git commit -m 'feat: implement task {state.task_id}'"

DO NOT just describe what you would do. ACTUALLY DO IT using the tools.
"""
        return self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

    def run_test(self, state: TaskState) -> AgentResult:
        """
        Run verification of DoD criteria.

        Strategy:
        1. Try the LLM-based test agent first (it should use tools to verify)
        2. If the LLM doesn't use tools (common with smaller models),
           fall back to running the verification commands directly ourselves
        """
        agent_config = self.config.get_agent("test")

        # First: try direct verification if we have verification commands
        # This is more reliable than depending on a 7B model to use tools
        if state.dod and state.dod.criteria:
            direct_results = self._run_direct_verification(state)
            if direct_results is not None:
                return direct_results

        # Fallback: LLM-based test agent
        system_prompt = self._load_prompt(agent_config.system_prompt_file or "prompts/test.txt")
        if not system_prompt:
            system_prompt = "You are the TEST agent. Verify each DoD criterion by running actual commands."

        dod_criteria_list = ""
        if state.dod and state.dod.criteria:
            for idx, criterion in enumerate(state.dod.criteria):
                cmd = criterion.verification_command or "Manual inspection required"
                dod_criteria_list += f"\n### criterion-{idx}: {criterion.description}\nVerification command: `{cmd}`\n"

        user_prompt = f"""## VERIFY EACH CRITERION BY RUNNING COMMANDS

For EACH criterion below, use run_command to execute the verification command.

## DoD Criteria
{dod_criteria_list}

Output results as:
```test-results
{{
    "criterion-0": {{"passed": true, "evidence": "output"}},
    "criterion-1": {{"passed": false, "evidence": "error"}}
}}
```
"""
        result = self._run_agent(agent_config, system_prompt, user_prompt, tools=TOOL_DEFINITIONS)

        if result.success:
            result.test_report = self._parse_test_results(result.output)
            if state.dod and state.dod.criteria:
                test_results = result.test_report
                for idx, criterion in enumerate(state.dod.criteria):
                    cid = f"criterion-{idx}"
                    if cid in test_results and test_results[cid].get("passed"):
                        criterion.passed = True
                        criterion.evidence = test_results[cid].get("evidence", "")
                        logger.info(f"DoD {cid} PASSED: {criterion.description[:60]}")
                    else:
                        criterion.passed = False
                        logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]}")

                passed_count = sum(1 for c in state.dod.criteria if c.passed)
                total_count = len(state.dod.criteria)
                logger.info(f"DoD final count: {passed_count}/{total_count} criteria passed")
                if passed_count < total_count:
                    result.success = False
                    result.error = f"{passed_count}/{total_count} DoD criteria passed"

        return result

    def _run_direct_verification(self, state: TaskState) -> Optional[AgentResult]:
        """
        Run DoD verification commands directly without LLM involvement.

        This is the reliable path — we execute each verification command
        ourselves and check exit codes. No LLM needed.
        """
        if not state.dod or not state.dod.criteria:
            return None

        # Check if we have any verification commands to run
        has_commands = any(c.verification_command for c in state.dod.criteria)
        if not has_commands:
            return None  # Fall back to LLM-based verification

        logger.info("Running direct verification (no LLM)...")

        # Check if a venv exists in the working dir — if so, prefix commands
        venv_path = self.working_dir / "venv" / "bin" / "activate"
        venv_prefix = f"source {venv_path} && " if venv_path.exists() else ""
        if venv_prefix:
            logger.debug("  Found venv, will activate for verification commands")

        test_report = {}
        all_evidence = []

        for idx, criterion in enumerate(state.dod.criteria):
            cid = f"criterion-{idx}"
            cmd = criterion.verification_command

            if not cmd:
                # No command — try to infer one from the description
                cmd = self._infer_verification_command(criterion.description)
                if cmd:
                    logger.debug(f"  Inferred verification command for {cid}: {cmd}")

            if not cmd:
                # Still no command — can't verify directly
                criterion.passed = False
                criterion.evidence = "No verification command provided"
                test_report[cid] = {"passed": False, "evidence": "No verification command"}
                logger.warning(f"DoD {cid} SKIPPED (no command): {criterion.description[:60]}")
                all_evidence.append(f"{cid}: SKIPPED — no verification command")
                continue

            # Fix common command issues:
            # 1. Replace bare `pytest` with venv-activated or `python3 -m pytest`
            if cmd.startswith("pytest ") or cmd == "pytest":
                if venv_prefix:
                    cmd = f"{venv_prefix}{cmd}"
                else:
                    cmd = cmd.replace("pytest", "python3 -m pytest", 1)

            # 2. Replace bare `python ` with `python3 `
            if cmd.startswith("python ") and not cmd.startswith("python3"):
                cmd = "python3" + cmd[6:]

            # Execute the verification command
            result = self.tool_executor.execute("run_command", {"command": cmd, "timeout": 30})

            # Check if command succeeded (EXIT_CODE: 0)
            passed = "EXIT_CODE: 0" in result
            criterion.passed = passed
            criterion.evidence = result[:500]

            test_report[cid] = {"passed": passed, "evidence": result[:500]}

            if passed:
                logger.info(f"DoD {cid} PASSED: {criterion.description[:60]}")
                all_evidence.append(f"{cid}: PASSED — {cmd}")
            else:
                logger.warning(f"DoD {cid} FAILED: {criterion.description[:60]}")
                all_evidence.append(f"{cid}: FAILED — {result[:200]}")

        passed_count = sum(1 for c in state.dod.criteria if c.passed)
        total_count = len(state.dod.criteria)
        logger.info(f"DoD final count: {passed_count}/{total_count} criteria passed")

        success = passed_count == total_count

        output = "\n".join(all_evidence)
        if success:
            output += "\n\n✅ ALL DOD CRITERIA PASSED"
        else:
            failed_ids = [f"criterion-{i}" for i, c in enumerate(state.dod.criteria) if not c.passed]
            output += f"\n\n❌ DOD VERIFICATION FAILED\nFailed: {', '.join(failed_ids)}"

        return AgentResult(
            success=success,
            output=output,
            error=None if success else f"{passed_count}/{total_count} DoD criteria passed",
            test_report=test_report
        )

    def _infer_verification_command(self, description: str) -> Optional[str]:
        """
        Try to infer a verification command from a DoD criterion description.

        Handles common patterns like:
        - "File X exists" → test -f X
        - "Function X returns Y" → python3 -c "from module import func; assert ..."
        - "Valid syntax" → python3 -m py_compile file.py
        """
        desc_lower = description.lower()

        # Pattern: file exists
        file_match = re.search(r'file\s+[`"]?(\w+\.py)[`"]?\s+exists', desc_lower)
        if file_match:
            filename = file_match.group(1)
            return f"test -f {filename}"

        # Pattern: valid syntax
        syntax_match = re.search(r'[`"]?(\w+\.py)[`"]?\s+.*valid\s+syntax', desc_lower)
        if not syntax_match:
            syntax_match = re.search(r'valid\s+(?:python\s+)?syntax.*[`"]?(\w+\.py)[`"]?', desc_lower)
        if syntax_match:
            filename = syntax_match.group(1)
            return f"python3 -m py_compile {filename}"

        # Pattern: function returns value
        func_match = re.search(r'function\s+[`"]?(\w+)\(\)[`"]?\s+returns?\s+[`"\']([^`"\']+)[`"\']', desc_lower)
        if func_match:
            func_name = func_match.group(1)
            expected = func_match.group(2)
            # Try to guess the module from the function name
            module = func_name  # Assume module name matches
            return f'python3 -c "from {module} import {func_name}; assert {func_name}() == \'{expected}\'"'

        # Pattern: "returns 'Hello, World!'" or similar with function name
        returns_match = re.search(r'[`"]?(\w+)\(\)[`"]?\s+returns?\s+.*["\']([^"\']+)["\']', description)
        if returns_match:
            func_name = returns_match.group(1)
            expected = returns_match.group(2)
            return f'python3 -c "from hello_world import {func_name}; assert {func_name}() == \'{expected}\'"'

        return None

    # ============================================================
    # Parsing helpers
    # ============================================================

    def _parse_dod_from_output(self, output: str) -> Optional[DoD]:
        """Parse Definition of Done from agent output."""
        dod = DoD()

        # Look for ```dod block
        dod_match = re.search(r'```dod\n(.*?)\n```', output, re.DOTALL)
        if not dod_match:
            dod_match = re.search(r'### Definition of Done\n(.*?)(?=###|\Z)', output, re.DOTALL)
        if not dod_match:
            # Also try ## Definition of Done
            dod_match = re.search(r'## Definition of Done\n(.*?)(?=##|\Z)', output, re.DOTALL)

        if not dod_match:
            return None

        dod_text = dod_match.group(1)

        # Also look for verification commands section
        verify_section = ""
        verify_match = re.search(r'(?:### |## )?Verification Commands?\n(.*?)(?=###|##|\Z)', output, re.DOTALL)
        if verify_match:
            verify_section = verify_match.group(1)

        # Extract individual commands from verification section
        verify_commands = re.findall(r'`([^`]+)`', verify_section)

        for line in dod_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            match = re.match(r'-\s*\[([ xX])\]\s*(.+)', line)
            if match:
                checked = match.group(1).lower() == 'x'
                full_text = match.group(2).strip()

                # Try to extract inline verification command: (verify: `command`)
                verify_cmd = None
                cmd_match = re.search(r'\(verify:\s*`([^`]+)`\)', full_text)
                if cmd_match:
                    verify_cmd = cmd_match.group(1)
                    # Remove the verify part from description
                    description = re.sub(r'\s*\(verify:\s*`[^`]+`\)', '', full_text).strip()
                else:
                    description = full_text.rstrip(':').strip()

                cid = dod.add(description, command=verify_cmd)
                if checked:
                    dod.mark_passed(cid)

        # If we didn't find inline commands, try to match from verify section by index
        if verify_commands:
            for idx, criterion in enumerate(dod.criteria):
                if not criterion.verification_command and idx < len(verify_commands):
                    criterion.verification_command = verify_commands[idx]

        return dod if dod.criteria else None

    def _parse_test_results(self, output: str) -> dict:
        """Parse test results JSON from agent output."""
        json_match = re.search(r'```(?:test-results|json)\n(\{.*?\})\n```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: check for keywords
        results = {}
        if "ALL DOD CRITERIA PASSED" in output:
            results["overall"] = {"passed": True}
        elif "VERIFICATION FAILED" in output:
            results["overall"] = {"passed": False}

        return results
