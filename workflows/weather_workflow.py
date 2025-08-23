"""
Weather workflow example - demonstrates how to add new workflows.
"""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from workflows import BaseWorkflow, WorkflowDependencies, WorkflowOutput

class WeatherInfo(BaseModel):
    """Model for weather information."""
    temperature: str = Field(description="Current temperature")
    condition: str = Field(description="Weather condition")
    location: str = Field(description="Location")

class WeatherWorkflow(BaseWorkflow):
    """Example workflow for weather information."""
    
    def __init__(self):
        super().__init__()
        self.description = "Get weather information"
        self.triggers = [
            "weather", "temperature", "forecast", "how's the weather",
            "what's the weather like", "weather today"
        ]
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent for weather workflow."""
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=WorkflowOutput,
            system_prompt=(
                "You are a weather assistant. When asked about weather, "
                "provide helpful information in a natural, conversational way. "
                "If you don't have real weather data, give a friendly response "
                "suggesting they check a weather app. Keep responses concise "
                "and suitable for voice output. "
                "IMPORTANT: Use the conversation history to understand context. "
                "If someone asks about temperature units (Celsius/Fahrenheit), "
                "remember their preference and use it in future responses."
            )
        )
        
        @self.agent.tool
        async def get_weather(ctx: RunContext[WorkflowDependencies], location: str = "current") -> WeatherInfo:
            """Get weather information for a location."""
            # Check conversation history for temperature unit preference
            use_celsius = False
            if ctx.deps.conversation_history:
                for msg in ctx.deps.conversation_history:
                    if "celsius" in msg.get("content", "").lower():
                        use_celsius = True
                        break
            
            # This is a mock implementation
            # In a real app, you'd integrate with a weather API like OpenWeatherMap
            temp_f = 72
            temp_c = int((temp_f - 32) * 5/9)
            
            temperature = f"{temp_c}°C" if use_celsius else f"{temp_f}°F"
            
            return WeatherInfo(
                temperature=temperature,
                condition="Partly cloudy",
                location=location if location != "current" else "your area"
            )
