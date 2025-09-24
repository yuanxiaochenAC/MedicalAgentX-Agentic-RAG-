import re
from pydantic import Field, model_validator
from typing import Optional, List

from ..core.logging import logger
from ..core.module import BaseModule
from ..core.base_config import Parameter
from ..models.base_model import BaseLLM
from .action import Action, ActionInput, ActionOutput
from ..prompts.agent_generator import AGENT_GENERATION_ACTION
from ..prompts.tool_calling import AGENT_GENERATION_TOOLS_PROMPT
from ..utils.utils import normalize_text

class AgentGenerationInput(ActionInput):
    """
    Input specification for the agent generation action.
    """

    goal: str = Field(description="A detailed statement of the workflow's goal, explaining the objectives the entire workflow aims to achieve")
    workflow: str = Field(description="An overview of the entire workflow, detailing all sub-tasks with their respective names, descriptions, inputs, and outputs")
    task: str = Field(description="A detailed JSON representation of the sub-task requiring agent generation. It should include the task's name, description, inputs, and outputs.")

    history: Optional[str] = Field(default=None, description="Optional field containing previously selected or generated agents.")
    suggestion: Optional[str] = Field(default=None, description="Optional suggestions to refine the generated agents.")
    existing_agents: Optional[str] = Field(default=None, description="Optional field containing the description of predefined agents, including each agent's name, role, and available actions.")
    tools: Optional[str] = Field(default=None, description="Optional field containing the description of tools that agents can use, including each tool's name and functionality.")


class GeneratedAgent(BaseModule):
    """
    Representation of a generated agent with validation capabilities.
    """

    name: str 
    description: str 
    inputs: List[Parameter]
    outputs: List[Parameter]
    prompt: str
    tool_names: Optional[List[str]] = None

    @classmethod
    def find_output_name(cls, text: str, outputs: List[str]):
        def sim(t1: str, t2: str):
            t1_words = normalize_text(t1).split()
            t2_words = normalize_text(t2).split()
            return len(set(t1_words)&set(t2_words))
        
        similarities = [sim(text, output) for output in outputs]
        max_sim = max(similarities)
        return outputs[similarities.index(max_sim)]

    @model_validator(mode="after")
    @classmethod
    def validate_prompt(cls, agent: 'GeneratedAgent'):
        """Validate and fix the agent's prompt template.
        
        This validator ensures that:
        1. All input parameters are properly referenced in the prompt
        2. Input references use the correct format with braces
        3. All output sections match the defined output parameters
        
        If there are mismatches in the output sections, it attempts to
        fix them by finding the most similar output name.
        
        Args:
            agent: The GeneratedAgent instance to validate.
            
        Returns:
            The validated and potentially modified GeneratedAgent.
            
        Raises:
            ValueError: If inputs are missing from the prompt or output sections don't match the defined outputs.
        """
        # check whether all the inputs are present in the prompt 
        input_names = [inp.name for inp in agent.inputs]
        prompt_has_inputs = [name in agent.prompt for name in input_names]
        if not all(prompt_has_inputs):
            missing_input_names = [name for name, has_input in zip(input_names, prompt_has_inputs) if not has_input]
            raise ValueError(f'The prompt miss inputs: {missing_input_names}')
        
        # check the format of the prompt to make sure it is wrapped in brackets. 
        pattern = r"### Instructions(.*?)### Output Format"
        prompt = agent.prompt

        def replace_with_braces(match):
            instructions = match.group(1)
            for name in input_names:
                instructions = re.sub(fr'<input>{{*\b{re.escape(name)}\b}}*</input>', fr'<input>{{{name}}}</input>', instructions)
            return "### Instructions" + instructions + "### Output Format"
        
        modified_prompt = re.sub(pattern, replace_with_braces, prompt, flags=re.DOTALL)
        agent.prompt = modified_prompt

        # check whether all the outputs are present in the prompt
        prompt = agent.prompt
        pattern = r"### Output Format(.*)"
        outputs_names = [out.name for out in agent.outputs]

        def fix_output_names(match):
            output_format = match.group(1)
            matches = re.findall(r"## ([^\n#]+)", output_format, flags=re.DOTALL)
            generated_outputs = [m.strip() for m in matches if m.strip() != "Thought"]
            # check the number of generated outputs and agent outputs 
            if len(generated_outputs) != len(outputs_names):
                raise ValueError(f"The number of outputs in the prompt is different from that defined in the `outputs` field of the agent. The outputs in the prompt are: {generated_outputs}, while the outputs from the agent's `outputs` field are: {outputs_names}")
            # check whether the generated output names are the same as agent outputs 
            for generated_output in generated_outputs:
                if generated_output not in outputs_names:
                    most_similar_output_name = cls.find_output_name(text=generated_output, outputs=outputs_names)
                    output_format = output_format.replace(generated_output, most_similar_output_name)
                    logger.warning(f"Couldn't find output name in prompt ('{generated_output}') in agent's outputs. Replace it with the most similar agent output: '{most_similar_output_name}'")
            return "### Output Format" + output_format
        
        modified_prompt = re.sub(pattern, fix_output_names, prompt, flags=re.DOTALL)
        agent.prompt = modified_prompt

        return agent


class AgentGenerationOutput(ActionOutput):

    selected_agents: List[str] = Field(description="A list of selected agent's names")
    generated_agents: List[GeneratedAgent] = Field(description="A list of generated agetns to address a sub-task")
    

class AgentGeneration(Action):
    """
    Action for generating agent specifications for workflow tasks.
    
    This action analyzes task requirements and generates appropriate agent
    specifications, including their prompts, inputs, and outputs. It can either
    select from existing agents or create new ones tailored to the task.
    """

    def __init__(self, **kwargs):
        name = kwargs.pop("name") if "name" in kwargs else AGENT_GENERATION_ACTION["name"]
        description = kwargs.pop("description") if "description" in kwargs else AGENT_GENERATION_ACTION["description"]
        prompt = kwargs.pop("prompt") if "prompt" in kwargs else AGENT_GENERATION_ACTION["prompt"]
        # inputs_format = kwargs.pop("inputs_format") if "inputs_format" in kwargs else AgentGenerationInput
        # outputs_format = kwargs.pop("outputs_format") if "outputs_format" in kwargs else AgentGenerationOutput
        inputs_format = kwargs.pop("inputs_format", None) or AgentGenerationInput
        outputs_format = kwargs.pop("outputs_format", None) or AgentGenerationOutput 
        tools = kwargs.pop("tools", None)
        super().__init__(name=name, description=description, prompt=prompt, inputs_format=inputs_format, outputs_format=outputs_format, **kwargs)
        self.tools = tools
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> AgentGenerationOutput:
        """Execute the agent generation process.
        
        This method uses the provided language model to generate agent specifications
        based on the workflow context and task requirements.
        
        Args:
            llm: The language model to use for generation.
            inputs: Input data containing workflow and task information.
            sys_msg: Optional system message for the language model.
            return_prompt: Whether to return both the generated agents and the prompt used.
            **kwargs: Additional keyword arguments.
            
        Returns:
            If return_prompt is False (default): The generated agents output.
            If return_prompt is True: A tuple of (generated agents, prompt used).
            
        Raises:
            ValueError: If the inputs are None or empty.
        """
        if not inputs:
            logger.error("AgentGeneration action received invalid `inputs`: None or empty.")
            raise ValueError('The `inputs` to AgentGeneration action is None or empty.')
        
        inputs_format: AgentGenerationInput = self.inputs_format
        outputs_format: AgentGenerationOutput = self.outputs_format

        prompt_params_names = inputs_format.get_attrs()
        prompt_params_values = {param: inputs.get(param, "") for param in prompt_params_names}
        if self.tools:
            tool_description = [
                {
                    tool.name: [
                        s["function"]["description"] for s in tool.get_tool_schemas()
                    ],
                }
                for tool in self.tools
            ]
            prompt_params_values["tools"] = AGENT_GENERATION_TOOLS_PROMPT.format(tools_description=tool_description)
        prompt = self.prompt.format(**prompt_params_values)
        agents = llm.generate(
            prompt = prompt, 
            system_message = sys_msg, 
            parser=outputs_format,
            parse_mode="json"
        )
        
        if return_prompt:
            return agents, prompt
        
        return agents
    