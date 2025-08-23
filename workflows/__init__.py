"""
Workflows package for LemmeTalk voice assistant using Pydantic AI.
Each workflow is a plugin that can be dynamically loaded.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import importlib
import os

@dataclass
class WorkflowDependencies:
    """Dependencies shared across all workflows."""
    user_id: str = "default"
    # Add more shared dependencies as needed

class WorkflowOutput(BaseModel):
    """Base output model for all workflows."""
    response: str = Field(description="The response to speak to the user")
    success: bool = Field(description="Whether the workflow executed successfully")
    workflow_name: str = Field(description="Name of the workflow that handled the request")

class BaseWorkflow:
    """Base class for all workflows using Pydantic AI."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "Base workflow"
        self.triggers = []  # Keywords that trigger this workflow
        self.agent: Optional[Agent] = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent for this workflow."""
        # This will be overridden by subclasses
        pass
    
    def can_handle(self, user_input: str) -> bool:
        """Check if this workflow can handle the user input."""
        user_input_lower = user_input.lower()
        return any(trigger.lower() in user_input_lower for trigger in self.triggers)
    
    async def execute(self, user_input: str, deps: WorkflowDependencies) -> WorkflowOutput:
        """Execute the workflow and return response."""
        if not self.agent:
            return WorkflowOutput(
                response="This workflow is not properly configured.",
                success=False,
                workflow_name=self.name
            )
        
        try:
            result = await self.agent.run(user_input, deps=deps)
            return WorkflowOutput(
                response=result.output.response,
                success=result.output.success,
                workflow_name=self.name
            )
        except Exception as e:
            return WorkflowOutput(
                response=f"Sorry, I encountered an error: {str(e)}",
                success=False,
                workflow_name=self.name
            )
    
    def get_help(self) -> str:
        """Return help text for this workflow."""
        return f"{self.name}: {self.description}"

class WorkflowManager:
    """Manages all available workflows using Pydantic AI."""
    
    def __init__(self):
        self.workflows: Dict[str, BaseWorkflow] = {}
        self._load_workflows()
    
    def _load_workflows(self):
        """Dynamically load all workflow modules."""
        workflows_dir = os.path.dirname(__file__)
        
        for filename in os.listdir(workflows_dir):
            if filename.endswith('_workflow.py'):
                module_name = filename[:-3]  # Remove .py
                try:
                    module = importlib.import_module(f"workflows.{module_name}")
                    
                    # Look for workflow classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseWorkflow) and 
                            attr != BaseWorkflow):
                            workflow = attr()
                            self.workflows[workflow.name] = workflow
                            print(f"Loaded workflow: {workflow.name}")
                            
                except Exception as e:
                    print(f"Failed to load workflow {module_name}: {e}")
    
    def get_workflow_for_input(self, user_input: str) -> Optional[BaseWorkflow]:
        """Find the best workflow for the given input."""
        for workflow in self.workflows.values():
            if workflow.can_handle(user_input):
                return workflow
        return None
    
    def list_workflows(self) -> List[str]:
        """List all available workflows."""
        return [workflow.get_help() for workflow in self.workflows.values()]
    
    def get_workflow(self, name: str) -> Optional[BaseWorkflow]:
        """Get a specific workflow by name."""
        return self.workflows.get(name)
