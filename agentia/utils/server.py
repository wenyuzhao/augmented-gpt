import yaml
from pathlib import Path

from agentia.agent import Agent
from agentia.plugins import ALL_PLUGINS

AGENTS_FOLDERS = [Path.cwd(), Path.cwd() / "agents"]


def __load_config_file(id: str):
    """Load a configuration file"""
    id = id.strip()
    has_ext = id.endswith(".yaml") or id.endswith(".yml")
    for folder in AGENTS_FOLDERS:
        if has_ext:
            file = folder / id
        else:
            file = folder / f"{id}.yaml"
            if not file.exists():
                file = folder / f"{id}.yml"
        if file.exists():
            return file, yaml.safe_load(file.read_text())
    raise FileNotFoundError(f"Agent not found: {id}")


def __load_agent_from_config(id: str, pending: set[str], agents: dict[str, Agent]):
    """Load a bot from a configuration file"""
    file, config = __load_config_file(id)
    id = config.get("id", id)
    if file.stem in pending:
        raise ValueError(f"Circular dependency detected: {id}")
    if file.stem in agents:
        return agents[file.stem]
    pending.add(file.stem)

    # Create tools
    tools = []
    if "tools" in config:
        if not isinstance(config["tools"], dict):
            raise ValueError("Invalid tools configuration: must be a dictionary")
        for name, c in config["tools"].items():
            if name not in ALL_PLUGINS:
                raise ValueError(f"Unknown tool: {name}")
            Plugin = ALL_PLUGINS[name]
            tools.append(Plugin(config=c or {}))

    # Load colleagues
    colleagues = []
    if "colleagues" in config:
        for colleague_id in config["colleagues"]:
            colleague = __load_agent_from_config(colleague_id, pending, agents)
            colleagues.append(colleague)

    agent = Agent(
        name=config.get("name"),
        description=config.get("description"),
        model=config.get("model"),
        tools=tools,
        instructions=config.get("instructions"),
        colleagues=colleagues,
    )
    pending.remove(file.stem)
    agents[file.stem] = agent
    return agent


def load_agent_from_config(id: str):
    """Load a bot from a configuration file"""
    return __load_agent_from_config(id, set(), dict())
