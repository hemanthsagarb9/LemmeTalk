"""
News workflow for reading Hacker News articles.
"""

import httpx
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from workflows import BaseWorkflow, WorkflowDependencies, WorkflowOutput

class NewsArticle(BaseModel):
    """Model for a news article."""
    title: str = Field(description="Article title")
    url: str = Field(description="Article URL")
    score: int = Field(description="Article score")
    author: str = Field(description="Article author")

class NewsOutput(BaseModel):
    """Output model for news workflow."""
    response: str = Field(description="The response to speak to the user")
    success: bool = Field(description="Whether the workflow executed successfully")
    articles: List[NewsArticle] = Field(description="List of articles found")

class NewsWorkflow(BaseWorkflow):
    """Workflow for reading Hacker News articles."""
    
    def __init__(self):
        super().__init__()
        self.description = "Read top Hacker News articles like a news bulletin"
        self.triggers = [
            "news", "hacker news", "hn", "top articles", "news bulletin",
            "read news", "latest news", "tech news"
        ]
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent for news workflow."""
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=NewsOutput,
            system_prompt=(
                "You are a news reader assistant. When given Hacker News articles, "
                "present them in a natural, conversational way as if reading a news bulletin. "
                "Focus on the most interesting stories and provide context. "
                "Keep responses concise but engaging for voice output."
            )
        )
        
        @self.agent.tool
        async def get_top_hn_articles(ctx: RunContext[WorkflowDependencies], count: int = 10) -> List[NewsArticle]:
            """Fetch top Hacker News articles."""
            async with httpx.AsyncClient() as client:
                # Get top story IDs
                response = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
                story_ids = response.json()[:count]
                
                articles = []
                for story_id in story_ids:
                    story_response = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
                    story = story_response.json()
                    
                    if story and 'title' in story:
                        articles.append(NewsArticle(
                            title=story.get('title', ''),
                            url=story.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                            score=story.get('score', 0),
                            author=story.get('by', 'Unknown')
                        ))
                
                return articles

class RemindersWorkflow(BaseWorkflow):
    """Workflow for managing reminders."""
    
    def __init__(self):
        super().__init__()
        self.description = "Add and manage reminders"
        self.triggers = [
            "reminder", "remind me", "add reminder", "set reminder",
            "remind", "todo", "task"
        ]
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent for reminders workflow."""
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=WorkflowOutput,
            system_prompt=(
                "You are a reminder assistant. Help users add reminders and tasks. "
                "Extract the task and any timing information from their request. "
                "Respond in a natural, conversational way suitable for voice output."
            )
        )
        
        @self.agent.tool
        async def add_reminder(ctx: RunContext[WorkflowDependencies], task: str, due_date: str = None) -> str:
            """Add a reminder to the user's list."""
            # In a real implementation, this would save to a database
            reminder_text = f"Reminder: {task}"
            if due_date:
                reminder_text += f" (due: {due_date})"
            
            # For now, just return a confirmation
            return f"Added reminder: {reminder_text}"

class ShoppingWorkflow(BaseWorkflow):
    """Workflow for managing shopping lists."""
    
    def __init__(self):
        super().__init__()
        self.description = "Add items to shopping list"
        self.triggers = [
            "shopping", "shopping list", "add to list", "buy", "purchase",
            "grocery", "shopping cart"
        ]
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent for shopping workflow."""
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=WorkflowOutput,
            system_prompt=(
                "You are a shopping list assistant. Help users add items to their shopping list. "
                "Extract the items they want to add and respond naturally. "
                "Keep responses conversational and suitable for voice output."
            )
        )
        
        @self.agent.tool
        async def add_to_shopping_list(ctx: RunContext[WorkflowDependencies], items: List[str]) -> str:
            """Add items to the shopping list."""
            # In a real implementation, this would save to a database
            items_text = ", ".join(items)
            return f"Added to shopping list: {items_text}"
