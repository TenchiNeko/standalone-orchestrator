"""
Configuration for the standalone orchestrator.

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
    max_tokens: int = 8192
    supports_tools: bool = True  # whether model handles tool_call format


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
    max_iterations: int = 10
    plan_dir: str = ".agents/plans"
    report_dir: str = ".agents/reports"
    log_dir: str = ".agents/logs"
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

    Adjust endpoints/models to match your infrastructure.
    """

    # --- Model definitions ---

    # Cortana GPU — 32B (primary workhorse)
    qwen_32b_cortana = ModelConfig(
        name="Qwen 2.5 Coder 32B (Cortana)",
        provider="ollama",
        endpoint="http://127.0.0.1:11435",
        model_id="qwen2.5-coder:32b",
        temperature=0.0,
        max_tokens=8192,
        supports_tools=True,
    )

    # Cortana GPU — 7B (lightweight, runs alongside 32B)
    qwen_7b_cortana = ModelConfig(
        name="Qwen 2.5 Coder 7B (Cortana)",
        provider="ollama",
        endpoint="http://127.0.0.1:11435",
        model_id="qwen2.5-coder:7b",
        temperature=0.0,
        max_tokens=8192,
        supports_tools=True,
    )

    # PVE node — 7B (isolated, handles explore + test)
    qwen_7b_pve = ModelConfig(
        name="Qwen 2.5 Coder 7B (PVE)",
        provider="ollama",
        endpoint="http://192.168.68.73:11434",
        model_id="qwen2.5-coder:7b",
        temperature=0.0,
        max_tokens=8192,
        supports_tools=True,
    )

    # --- Agent role assignments ---
    # 32B on Cortana: plan + build (sequential, share GPU)
    # 7B on Cortana: initializer (runs once at start, then frees VRAM)
    # 7B on PVE: explore + test (offloaded, no Cortana contention)
    agents = {
        "initializer": AgentConfig(
            role="initializer",
            model=qwen_7b_cortana,
            system_prompt_file="prompts/initializer.txt",
            timeout_seconds=300,
        ),
        "explore": AgentConfig(
            role="explore",
            model=qwen_7b_pve,
            system_prompt_file="prompts/explore.txt",
            timeout_seconds=300,
        ),
        "plan": AgentConfig(
            role="plan",
            model=qwen_32b_cortana,
            system_prompt_file="prompts/plan.txt",
            timeout_seconds=600,
        ),
        "build": AgentConfig(
            role="build",
            model=qwen_32b_cortana,
            system_prompt_file="prompts/build.txt",
            timeout_seconds=900,
            max_tool_rounds=80,
        ),
        "test": AgentConfig(
            role="test",
            model=qwen_7b_pve,
            system_prompt_file="prompts/test.txt",
            timeout_seconds=600,
            max_tool_rounds=40,
        ),
    }

    return Config(max_iterations=10, agents=agents)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load config from JSON file, falling back to defaults."""
    config = default_config()

    if config_path and config_path.exists():
        with open(config_path) as f:
            overrides = json.load(f)

        if "max_iterations" in overrides:
            config.max_iterations = overrides["max_iterations"]

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
                    if "max_tool_rounds" in ao:
                        agent.max_tool_rounds = ao["max_tool_rounds"]

    return config
