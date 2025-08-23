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


