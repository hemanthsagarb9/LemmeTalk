# LemmeTalk - Voice Assistant with Pluggable Workflows

A powerful voice assistant built with Pydantic AI that supports pluggable workflows for different tasks.

## Features

- 🎤 **Voice Interface**: Natural voice interaction using Kokoro TTS
- 🧠 **AI-Powered**: Built on Pydantic AI for type-safe, structured responses
- 🔌 **Pluggable Workflows**: Easy to add new capabilities
- 🎯 **Smart Routing**: Automatically routes requests to appropriate workflows
- 🗣️ **TTS Optimized**: Responses optimized for natural speech

## Architecture

```
lemmtalk/
├── voice_loop.py          # Main orchestrator
├── workflows/             # Plugin directory
│   ├── __init__.py        # Workflow management system
│   ├── news_workflow.py   # Hacker News reader
│   ├── reminders_workflow.py # Task management
│   ├── shopping_workflow.py # Shopping list
│   └── weather_workflow.py # Weather information
├── tools/                 # Utility tools
│   ├── __init__.py        # Tools package
│   └── storage.py         # File storage system
├── data/                  # Persistent data storage
│   ├── shopping_list.json # Shopping list data
│   └── reminders.json     # Reminders data
└── config/               # Configuration
```

## Available Workflows

### 1. News Workflow
**Triggers**: "news", "hacker news", "hn", "top articles", "news bulletin"
- Fetches top 5 Hacker News articles with real-time data
- Extracts brief content previews for key articles
- Creates concise, conversational summaries
- Presents stories in natural, voice-optimized format
- Includes article scores and author information

### 2. Reminders Workflow
**Triggers**: "reminder", "remind me", "add reminder", "todo", "task"
- Adds reminders and tasks with persistent storage
- Extracts timing information from natural language
- Manages personal task list
- View current reminders: "What are my reminders?"
- Mark as completed: "Mark reminder X as done"
- Clear completed reminders

### 3. Shopping Workflow
**Triggers**: "shopping", "shopping list", "add to list", "buy", "grocery"
- Manages shopping lists with persistent storage
- Adds items from voice commands
- Organizes grocery items
- View current list: "What's on my shopping list?"
- Mark items completed: "Mark milk as bought"
- Clear completed items

## Installation

1. **Clone and setup environment**:
```bash
git clone <repository>
cd lemmtalk
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install dependencies**:
```bash
pip install pydantic-ai kokoro faster-whisper sounddevice soundfile python-dotenv openai httpx
```

3. **Setup environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

Run the voice assistant:
```bash
python voice_loop.py
```

### Example Commands

- **"Read me the top news"** → Fetches top 5 HN articles with concise summaries
- **"Remind me to call mom tomorrow"** → Adds a reminder
- **"What are my reminders?"** → Shows current reminders
- **"Add milk and bread to my shopping list"** → Updates shopping list
- **"What's on my shopping list?"** → Shows current items
- **"Mark milk as bought"** → Marks item as completed
- **"What's the weather like?"** → General conversation (fallback)

## Adding New Workflows

1. **Create a new workflow file** in `workflows/`:
```python
# workflows/youtube_workflow.py
from workflows import BaseWorkflow, WorkflowDependencies, WorkflowOutput
from pydantic_ai import Agent, RunContext

class YouTubeWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__()
        self.description = "Play YouTube videos"
        self.triggers = ["youtube", "play video", "watch"]
    
    def _setup_agent(self):
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=WorkflowOutput,
            system_prompt="You are a YouTube assistant..."
        )
        
        @self.agent.tool
        async def play_video(ctx: RunContext[WorkflowDependencies], url: str):
            # Implementation here
            pass
```

2. **The workflow is automatically loaded** when you restart the assistant

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4o-mini)
- `KOKORO_SPEAKER`: Voice to use (default: af_heart)
- `KOKORO_SPEED`: Speech speed (default: 1.2)

### Voice Options

- `af_heart` - High quality (default)
- `af_sky` - Good quality
- `af_alloy` - Alternative female
- `af_nicole` - Another female option
- `am_michael` - Male voice

## Technical Details

### Pydantic AI Integration

The system uses [Pydantic AI](https://ai.pydantic.dev) for:
- **Type-safe workflows** with Pydantic models
- **Structured outputs** for consistent responses
- **Built-in tool system** for external integrations
- **Dependency injection** for shared services

### Workflow System

- **Dynamic loading**: Workflows are automatically discovered
- **Keyword matching**: Triggers determine which workflow handles requests
- **Fallback system**: General conversation when no workflow matches
- **Async support**: All workflows support async operations

### TTS Optimization

- **Natural speech patterns**: Avoids reading numbers and symbols
- **Conversational flow**: Uses "first", "second", "third" instead of "1, 2, 3"
- **Clean formatting**: Removes markdown and technical notation
- **Speed control**: Adjustable speech rate

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your workflow in `workflows/`
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
