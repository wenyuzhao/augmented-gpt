from typing import Any
import yaml
from pathlib import Path

from agentia.agent import Agent
from agentia.plugins import ALL_PLUGINS, Plugin

AGENTS_SEARCH_PATHS = [
    Path.cwd(),
    Path.cwd() / "agents",
    Path.cwd() / ".agents",
    Path.home() / ".config" / "agentia" / "agents",
]


def __get_config_path(cwd: Path, id: str):
    id = id.strip()
    possible_paths = []
    if id.endswith(".yaml") or id.endswith(".yml"):
        possible_paths.append(id)
    else:
        possible_paths.extend([f"{id}.yaml", f"{id}.yml"])
    for s in possible_paths:
        p = Path(s)
        if not p.is_absolute():
            p = cwd / p
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Agent config not found: {id}")


def __create_tools(
    config: dict[str, Any], parent_tool_configs: dict[str, Any]
) -> tuple[list[Plugin], dict[str, Any]]:
    tools: list[Plugin] = []
    tool_configs = {}
    if "tools" in config:
        if not isinstance(config["tools"], dict):
            raise ValueError("Invalid tools configuration: must be a dictionary")
        for name, c in config["tools"].items():
            if name not in ALL_PLUGINS:
                raise ValueError(f"Unknown tool: {name}")
            PluginCls = ALL_PLUGINS[name]
            if not (c is None or isinstance(c, dict)):
                raise ValueError(
                    f"Invalid config for tool {name}: must be a dict or null"
                )
            if c is None and name in parent_tool_configs:
                c = parent_tool_configs[name]
            c = c or {}
            tool_configs[name] = c
            tools.append(PluginCls(config=c))
    return tools, tool_configs


def __load_agent_from_config(
    file: Path,
    pending: set[Path],
    agents: dict[Path, Agent],
    parent_tool_configs: dict[str, Any],
):
    """Load a bot from a configuration file"""
    # Load the configuration file
    assert file.exists()
    file = file.resolve()
    config = yaml.safe_load(file.read_text())
    # Already loaded?
    if file in pending:
        raise ValueError(f"Circular dependency detected: {file.stem}")
    pending.add(file)
    if file in agents:
        return agents[file]
    # Create tools
    tools, tool_configs = __create_tools(config, parent_tool_configs)
    # Load colleagues
    colleagues: list[Agent] = []
    if "colleagues" in config:
        for child_id in config["colleagues"]:
            child_path = __get_config_path(file.parent, child_id)
            colleague = __load_agent_from_config(
                child_path, pending, agents, tool_configs
            )
            colleagues.append(colleague)
    # Create agent
    knowledge_base: str | bool = config.get("knowledge_base", False)
    agent_id = config.get("id", file.stem)
    agent = Agent(
        name=config.get("name"),
        id=agent_id,
        icon=config.get("icon"),
        description=config.get("description"),
        model=config.get("model"),
        tools=tools,
        instructions=config.get("instructions"),
        colleagues=colleagues,
        knowledge_base=(
            Path(knowledge_base) if isinstance(knowledge_base, str) else knowledge_base
        ),
    )
    agent.original_config = config
    pending.remove(file)
    agents[file] = agent
    return agent


def load_agent_from_config(name: str | Path) -> Agent:
    """Load a bot from a configuration file"""
    if isinstance(name, Path):
        if not name.exists():
            raise FileNotFoundError(f"Agent config not found: {name}")
        config_path = name.resolve()
    elif name.endswith(".yaml") or name.endswith(".yml"):
        # name is also a path
        config_path = Path(name)
        if not config_path.exists() or not config_path.is_file():
            raise FileNotFoundError(f"Agent config not found: {name}")
        config_path = config_path.resolve()
    elif (s := Path(name).suffix) and s not in [".yaml", ".yml"]:
        raise ValueError(f"Invalid agent path: {name}")
    else:
        # If the name is a string, we need to find the configuration file from a list of search paths
        config_path = None
        for dir in AGENTS_SEARCH_PATHS:
            if (file := dir / f"{name}.yaml").exists():
                config_path = file
                break
            if (file := dir / f"{name}.yml").exists():
                config_path = file
                break
        if config_path is None:
            raise FileNotFoundError(f"Agent config not found: {name}")
    return __load_agent_from_config(config_path, set(), {}, {})
