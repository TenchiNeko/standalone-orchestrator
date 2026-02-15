"""
Configuration for the standalone orchestrator.

v0.8.0: Dual Ollama instance architecture.
  - Instance 1 (port 11435): Llama 3.3 70B on GPUs 1,2,3 — plan + build
  - Instance 2 (port 11436): Qwen 2.5 Coder 14B on GPU 0 (5060 Ti) — init, explore, test
  - No model swapping — both run simultaneously
  - RAG Knowledge Base on port 8787

Defines model routing via direct Ollama/API HTTP endpoints.
No CLI tool dependencies.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ModelConfig:
    """Configuration for a model endpoint."""
    name: str
    provider: str  # "ollama" or "anthropic"
    model_id: str
    endpoint: Optional[str] = None  # HTTP base URL for ollama
    api_key_env: Optional[str] = None  # env var name for API key
    temperature: float = 0.0
    max_tokens: int = 16384
    context_window: int = 131072  # total context window in tokens (default 128K for Llama 3.3)
    supports_tools: bool = True  # whether model handles tool_call format
    native_tool_calling: bool = False  # True = model returns structured tool_calls natively (Qwen3, Llama 3.3)
    tool_call_style: str = "text"  # "text" = parse from content, "native" = use Ollama tool_calls field

    # v1.2: Inference optimization fields
    repeat_penalty: float = 1.0  # 1.0=disabled. Qwen-Next is very sensitive to penalties (Stepfunction/r/LocalLLaMA)
    thinking_mode: str = "auto"  # "enabled"=always /think, "disabled"=always /no_think, "auto"=per-agent
    thinking_budget: int = 0  # max thinking tokens (0=unlimited). Qwen3 native budget control
    num_keep: int = -1  # tokens to keep from initial prompt for KV cache (-1=auto)
    top_p: float = 0.95  # nucleus sampling (Qwen recommended)
    min_p: float = 0.0  # min-p sampling (0=disabled)
    draft_model: Optional[str] = None  # speculative decoding draft model (Ollama server-side config)


@dataclass
class AgentConfig:
    """Configuration for a specific agent role."""
    role: str
    model: ModelConfig
    system_prompt_file: Optional[str] = None
    timeout_seconds: int = 600
    retry_count: int = 2
    max_tool_rounds: int = 50  # max tool-use round trips per invocation


@dataclass
class Config:
    """Main configuration container."""
    max_iterations: int = 3
    plan_dir: str = ".agents/plans"
    report_dir: str = ".agents/reports"
    log_dir: str = ".agents/logs"
    kb_url: str = "http://localhost:8787"  # v0.8.0: RAG Knowledge Base server
    librarian_model: Optional['ModelConfig'] = None  # v0.9.0: Librarian curator model
    agents: Dict[str, AgentConfig] = field(default_factory=dict)

    def get_agent(self, role: str) -> AgentConfig:
        if role not in self.agents:
            raise ValueError(f"Unknown agent role: {role}")
        return self.agents[role]

    @staticmethod
    def load_default() -> "Config":
        """Create default config for local Ollama setup."""
        return default_config()


def default_config() -> Config:
    """
    Default configuration using local Ollama endpoints.

    v0.8.0 Dual-Instance Architecture:
    ===================================
    Cortana (Dell 7920): 4x GPU — 64GB VRAM total
      Instance 1 (port 11435): Llama 3.3 70B (~40GB VRAM)
        - GPUs: 1 (RTX 3090 24GB) + 2 (RTX 4070 Super 12GB) + 3 (RTX 4070 Super 12GB)
        - Roles: plan, build (heavy reasoning)
        - CUDA_VISIBLE_DEVICES=1,2,3

      Instance 2 (port 11436): Qwen 2.5 Coder 14B (~9GB VRAM)
        - GPU: 0 (RTX 5060 Ti 16GB)
        - Roles: initializer, explore, test (fast, lightweight)
        - CUDA_VISIBLE_DEVICES=0

    Both instances run simultaneously — NO model swapping.
    The 5060 Ti was previously idle during 70B inference.

    RAG Knowledge Base: port 8787 (systemd service: rag-kb)
      - Tier 1: 60+ error→solution patterns
      - Tier 2: 15,500+ documentation chunks (CPython, Flask, pytest, etc.)

    Optional PVE Node (fallback if Cortana GPU 0 is needed):
      - Endpoint: http://192.168.68.73:11434
      - Model: Qwen 2.5 Coder 7B
    """

    # --- Model definitions ---

    # PRIMARY (Instance 1, port 11435): Llama 3.3 70B — heavy reasoning
    llama_70b = ModelConfig(
        name="Llama 3.3 70B (Cortana Instance 1)",
        provider="ollama",
        endpoint=os.environ.get("OLLAMA_PRIMARY_URL", "http://127.0.0.1:11435"),
        model_id="qwen3-coder-next",
        temperature=0.0,
        max_tokens=16384,
        context_window=131072,  # 128K context
        supports_tools=True,
        # v1.2: Inference optimizations
        repeat_penalty=1.0,  # DISABLED — Qwen-Next very sensitive (Stepfunction/r/LocalLLaMA)
        thinking_mode="auto",  # Per-agent: enabled for plan/build, disabled for fast tasks
        thinking_budget=0,  # 0=unlimited thinking for heavy reasoning
        top_p=0.95,
    )

    # SECONDARY (Instance 2, port 11436): Qwen 2.5 Coder 14B — fast agent work
    # Runs on dedicated GPU 0 (5060 Ti 16GB) — always loaded, no swapping
    qwen_14b = ModelConfig(
        name="Qwen 2.5 Coder 14B (Cortana Instance 2)",
        provider="ollama",
        endpoint=os.environ.get("OLLAMA_SECONDARY_URL", "http://100.81.200.82:11434"),
        model_id="qwen2.5-coder:7b",
        temperature=0.0,
        max_tokens=16384,
        context_window=32768,  # 32K context
        supports_tools=True,
        # v1.2: Inference optimizations
        repeat_penalty=1.0,  # DISABLED for Qwen family
        thinking_mode="disabled",  # Fast agent — no extended reasoning
    )

    qwen_14b_testgen = ModelConfig(
        name="Qwen 2.5 Coder 14B LoRA Test-Gen (Cortana Instance 2)",
        provider="ollama",
        endpoint=os.environ.get("OLLAMA_SECONDARY_URL", "http://100.81.200.82:11434"),
        model_id="qwen-coder-testgen:14b",
        temperature=0.3,
        max_tokens=16384,
        context_window=32768,
        supports_tools=True,
        # v1.2: Inference optimizations
        repeat_penalty=1.0,
        thinking_mode="disabled",  # Test gen doesn't need extended reasoning
    )

    # FALLBACK: PVE Node 7B (if GPU 0 is reclaimed for 70B)
    # qwen_7b_pve = ModelConfig(
    #     name="Qwen 2.5 Coder 7B (PVE Node)",
    #     provider="ollama",
    #     endpoint="http://192.168.68.73:11434",
    #     model_id="qwen2.5-coder:7b",
    #     temperature=0.0,
    #     max_tokens=16384,
    #     context_window=32768,
    #     supports_tools=True,
    # )

    # --- Qwen3 alternatives (native tool calling) ---
    # Uncomment and swap into agent assignments when available.

    # Qwen3 30B-A3B (MoE, only 3B active)
    # qwen3_30b = ModelConfig(
    #     name="Qwen3 30B-A3B (Cortana)",
    #     provider="ollama",
    #     endpoint=os.environ.get("OLLAMA_PRIMARY_URL", "http://127.0.0.1:11435"),
    #     model_id="qwen3:30b-a3b",
    #     temperature=0.0,
    #     max_tokens=16384,
    #     context_window=131072,
    #     supports_tools=True,
    #     native_tool_calling=True,
    #     tool_call_style="native",
    # )

    # --- Agent role assignments ---
    # Instance 1 (70B, port 11435): plan + build — best reasoning for code generation
    # Instance 2 (14B, port 11436): initializer + explore + test — fast, co-located

    # v0.9.0: Librarian model — uses Instance 2 (same as init/explore/test)
    # Runs post-session curation: error patterns, journal entries, code snippets.
    # Can also point to PVE node 7B to offload: http://192.168.68.73:11434
    librarian_model = ModelConfig(
        name="Qwen 2.5 Coder 14B (Librarian)",
        provider="ollama",
        endpoint=os.environ.get("OLLAMA_SECONDARY_URL", "http://100.81.200.82:11434"),
        model_id="qwen2.5-coder:7b",
        temperature=0.1,
        max_tokens=4096,
        context_window=32768,
        supports_tools=False,
    )

    agents = {
        "initializer": AgentConfig(
            role="initializer",
            model=qwen_14b,
            system_prompt_file="prompts/initializer.txt",
            timeout_seconds=300,
        ),
        "explore": AgentConfig(
            role="explore",
            model=qwen_14b,
            system_prompt_file="prompts/explore.txt",
            timeout_seconds=300,
        ),
        "plan": AgentConfig(
            role="plan",
            model=llama_70b,
            system_prompt_file="prompts/plan.txt",
            timeout_seconds=900,
        ),
        "build": AgentConfig(
            role="build",
            model=llama_70b,
            system_prompt_file="prompts/build.txt",
            timeout_seconds=900,
            max_tool_rounds=25,
        ),
        "test": AgentConfig(
            role="test",
            model=qwen_14b,
            system_prompt_file="prompts/test.txt",
            timeout_seconds=600,
            max_tool_rounds=40,
        ),
        "test_gen": AgentConfig(
            role="test_gen",
            model=qwen_14b_testgen,
            system_prompt_file="prompts/test_gen.txt",
            timeout_seconds=600,
            max_tool_rounds=25,
        ),
    }

    return Config(
        max_iterations=3,
        kb_url=os.environ.get("KB_URL", "http://localhost:8787"),
        librarian_model=librarian_model,
        agents=agents,
    )


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load config from JSON file, falling back to defaults."""
    config = default_config()

    if config_path and config_path.exists():
        with open(config_path) as f:
            overrides = json.load(f)

        if "max_iterations" in overrides:
            config.max_iterations = overrides["max_iterations"]

        if "kb_url" in overrides:
            config.kb_url = overrides["kb_url"]

        if "agents" in overrides:
            for role, ao in overrides["agents"].items():
                if role in config.agents:
                    agent = config.agents[role]
                    if "model" in ao:
                        agent.model.model_id = ao["model"]
                    if "provider" in ao:
                        agent.model.provider = ao["provider"]
                    if "endpoint" in ao:
                        agent.model.endpoint = ao["endpoint"]
                    if "timeout" in ao:
                        agent.timeout_seconds = ao["timeout"]
                    if "supports_tools" in ao:
                        agent.model.supports_tools = ao["supports_tools"]
                    if "native_tool_calling" in ao:
                        agent.model.native_tool_calling = ao["native_tool_calling"]
                    if "tool_call_style" in ao:
                        agent.model.tool_call_style = ao["tool_call_style"]
                    if "max_tool_rounds" in ao:
                        agent.max_tool_rounds = ao["max_tool_rounds"]
                    if "context_window" in ao:
                        agent.model.context_window = ao["context_window"]
                    # v1.2: Inference optimization overrides
                    if "repeat_penalty" in ao:
                        agent.model.repeat_penalty = ao["repeat_penalty"]
                    if "thinking_mode" in ao:
                        agent.model.thinking_mode = ao["thinking_mode"]
                    if "thinking_budget" in ao:
                        agent.model.thinking_budget = ao["thinking_budget"]
                    if "top_p" in ao:
                        agent.model.top_p = ao["top_p"]
                    if "min_p" in ao:
                        agent.model.min_p = ao["min_p"]

    return config
