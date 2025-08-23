"""
News workflow for reading Hacker News articles with podcast-style summaries.
"""

import httpx
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from workflows import BaseWorkflow, WorkflowDependencies, WorkflowOutput
import re
from bs4 import BeautifulSoup
import time

class NewsArticle(BaseModel):
    """Model for a news article."""
    title: str = Field(description="Article title")
    url: str = Field(description="Article URL")
    score: int = Field(description="Article score")
    author: str = Field(description="Article author")
    summary: Optional[str] = Field(description="Podcast-style summary", default=None)
    content_preview: Optional[str] = Field(description="Content preview", default=None)

class NewsOutput(BaseModel):
    """Output model for news workflow."""
    response: str = Field(description="The response to speak to the user")
    success: bool = Field(description="Whether the workflow executed successfully")
    articles: List[NewsArticle] = Field(description="List of articles found")
    podcast_summary: str = Field(description="Complete podcast-style news summary")

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
                "You are a podcast-style news reader for Hacker News. Create engaging, "
                "conversational summaries of tech news articles. Write as if you're hosting "
                "a tech news podcast. Use natural speech patterns, avoid reading numbers, "
                "and make the content engaging for voice output. "
                "Focus on the most interesting aspects of each story and provide context. "
                "Keep each article summary concise but informative."
            )
        )
        
        @self.agent.tool
        async def get_top_hn_articles_with_summaries(ctx: RunContext[WorkflowDependencies], count: int = 10) -> List[NewsArticle]:
            """Fetch top Hacker News articles with content extraction and summaries."""
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get top story IDs
                response = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
                story_ids = response.json()[:count]
                
                articles = []
                for i, story_id in enumerate(story_ids):
                    try:
                        # Get story details
                        story_response = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
                        story = story_response.json()
                        
                        if not story or 'title' not in story:
                            continue
                        
                        article = NewsArticle(
                            title=story.get('title', ''),
                            url=story.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                            score=story.get('score', 0),
                            author=story.get('by', 'Unknown')
                        )
                        
                        # Extract content preview if it's a web article
                        if article.url and not article.url.startswith('https://news.ycombinator.com'):
                            try:
                                content_preview = await self._extract_article_content(client, article.url)
                                article.content_preview = content_preview
                            except Exception as e:
                                print(f"Failed to extract content for {article.url}: {e}")
                        
                        articles.append(article)
                        
                        # Small delay to be respectful to servers
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Error processing story {story_id}: {e}")
                        continue
                
                return articles
        
        @self.agent.tool
        async def create_podcast_summary(ctx: RunContext[WorkflowDependencies], articles: List[NewsArticle]) -> str:
            """Create a podcast-style summary of the articles."""
            if not articles:
                return "No articles found to summarize."
            
            # Prepare article data for summary
            article_data = []
            for i, article in enumerate(articles, 1):
                data = f"Article {i}: {article.title}"
                if article.content_preview:
                    data += f" - Content: {article.content_preview[:200]}..."
                if article.score > 0:
                    data += f" - Score: {article.score}"
                article_data.append(data)
            
            # Use OpenAI to create podcast summary
            from openai import OpenAI
            import os
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = f"""Create a podcast-style news summary for these Hacker News articles. 
            Write as if you're hosting a tech news podcast. Use natural speech patterns, 
            avoid reading numbers, and make it engaging for voice output.
            
            Articles:
            {chr(10).join(article_data)}
            
            Format the response as a conversational podcast intro and then cover each story briefly. 
            Keep it engaging and natural for voice output."""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a tech news podcast host. Create engaging, conversational summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"Error creating podcast summary: {e}")
                # Fallback summary
                return f"Here are the top stories from Hacker News today. {chr(10).join([f'{i+1}. {article.title}' for i, article in enumerate(articles[:5])])}"
    
    async def _extract_article_content(self, client: httpx.AsyncClient, url: str) -> str:
        """Extract content preview from article URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', 'main', '.content', '.post-content', '.entry-content',
                '.article-content', '.story-content', '[role="main"]'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                # Fallback to body
                content = soup.find('body')
            
            if content:
                # Get text and clean it up
                text = content.get_text()
                # Remove extra whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                # Take first 300 characters
                return text[:300] + "..." if len(text) > 300 else text
            
            return ""
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""


