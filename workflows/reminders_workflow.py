"""
Reminders workflow for managing tasks and reminders.
"""

from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from workflows import BaseWorkflow, WorkflowDependencies, WorkflowOutput
from tools.storage import RemindersStorage

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
        self.storage = RemindersStorage()
        
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDependencies,
            output_type=WorkflowOutput,
            system_prompt=(
                "You are a reminder assistant. Help users add reminders and tasks, "
                "view their current reminders, and mark them as completed. "
                "Extract the task and any timing information from their request. "
                "Respond in a natural, conversational way suitable for voice output."
            )
        )
        
        @self.agent.tool
        async def add_reminder(ctx: RunContext[WorkflowDependencies], task: str, due_date: str = None) -> str:
            """Add a reminder to the user's list."""
            self.storage.add_reminder(task, due_date)
            reminder_text = f"Reminder: {task}"
            if due_date:
                reminder_text += f" (due: {due_date})"
            return f"Added reminder: {reminder_text}"
        
        @self.agent.tool
        async def get_reminders(ctx: RunContext[WorkflowDependencies]) -> str:
            """Get the current reminders."""
            reminders = self.storage.get_reminders()
            if not reminders:
                return "You have no reminders set."
            
            active_reminders = [r for r in reminders if not r.get('completed', False)]
            if not active_reminders:
                return "All your reminders have been completed!"
            
            reminder_texts = []
            for reminder in active_reminders:
                text = reminder['task']
                if 'due_date' in reminder:
                    text += f" (due: {reminder['due_date']})"
                reminder_texts.append(text)
            
            return f"Your active reminders: {', '.join(reminder_texts)}"
        
        @self.agent.tool
        async def mark_reminder_completed(ctx: RunContext[WorkflowDependencies], task_name: str) -> str:
            """Mark a reminder as completed."""
            self.storage.mark_completed(task_name)
            return f"Marked reminder '{task_name}' as completed."
        
        @self.agent.tool
        async def clear_completed_reminders(ctx: RunContext[WorkflowDependencies]) -> str:
            """Remove completed reminders."""
            self.storage.clear_completed()
            return "Cleared all completed reminders."
