# Agentia: Ergonomic LLM Agent Augmented with Tools


## Getting Started

```python
from agentia import Agent
from typing import Annotated

# Define a tool as a python function
def get_weather(location: Annotated[str, "The city name"]):
    """Get the current weather in a given location"""
    return { "temperature": 72 }

# Create an agent
agent = Agent(tools=[get_weather])

# Run the agent with the tool
response = await agent.chat_completion("What is the weather like in boston?")

print(response)

# Output: The current temperature in Boston is 72°F.
```

## Create an Agent from a Config File

1. Create a config file at `./alice.yml`

```yaml
name: Alice
icon: 👩
instructions: You are a helpful assistant
tools:
  clock:
  calculator:
  # ... other tools
```

2. In your python code:

```python
agent = Agent.load_from_config("./alice.yml")
```

3. Alternatively, start a REPL:

```bash
pipx install agentia
agentia repl alice
```