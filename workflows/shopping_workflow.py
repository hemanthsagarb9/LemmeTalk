"""
Shopping workflow for managing shopping lists.
"""

from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from workflows import BaseWorkflow, WorkflowDependencies, WorkflowOutput
from tools.storage import ShoppingListStorage

class ShoppingWorkflow(BaseWorkflow):
    """Workflow for managing shopping lists."""
    
    def __init__(self):
        super().__init__()
        self.description = "Add items to shopping list"
        self.triggers = [
            "shopping", "shopping list", "add to list", "buy", "purchase",
            "grocery", "shopping cart", "add", "list", "items"
        ]
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent for shopping workflow."""
        self.storage = ShoppingListStorage()
        
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=WorkflowOutput,
            system_prompt=(
                "You are a shopping list assistant. Help users add items to their shopping list, "
                "view their current list, and mark items as completed. "
                "Extract the items they want to add from natural language requests. "
                "For example, 'add eggs and red and chicken' should extract ['eggs', 'bread', 'chicken']. "
                "Keep responses conversational and suitable for voice output."
            )
        )
        
        @self.agent.tool
        async def add_to_shopping_list(ctx: RunContext[WorkflowDependencies], items: List[str]) -> str:
            """Add items to the shopping list."""
            self.storage.add_items(items)
            items_text = ", ".join(items)
            return f"Added to shopping list: {items_text}"
        
        @self.agent.tool
        async def get_shopping_list(ctx: RunContext[WorkflowDependencies]) -> str:
            """Get the current shopping list."""
            items = self.storage.get_items()
            if not items:
                return "Your shopping list is empty."
            
            active_items = [item for item in items if not item.get('completed', False)]
            if not active_items:
                return "All items on your shopping list have been completed!"
            
            item_names = [item['item'] for item in active_items]
            return f"Your shopping list has: {', '.join(item_names)}"
        
        @self.agent.tool
        async def mark_item_completed(ctx: RunContext[WorkflowDependencies], item_name: str) -> str:
            """Mark an item as completed."""
            self.storage.mark_completed(item_name)
            return f"Marked '{item_name}' as completed."
        
        @self.agent.tool
        async def clear_completed_items(ctx: RunContext[WorkflowDependencies]) -> str:
            """Remove completed items from the list."""
            self.storage.clear_completed()
            return "Cleared all completed items from your shopping list."
