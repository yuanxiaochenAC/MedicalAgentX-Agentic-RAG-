from .utils import extract_function_call_json, evaluate_function_call

class BaseTool:
    def __init__(self, tool_config):
        self.tool_config = tool_config

    def run(self):
        pass

    def check_function_call(self, function_call_json):
        if isinstance(function_call_json, str):
            function_call_json = extract_function_call_json(function_call_json)
        if function_call_json is not None:
            return evaluate_function_call(self.tool_config, function_call_json)
        else:
            return False, "Invalid JSON string of function call"

    def get_required_parameters(self):
        """
        Retrieve required parameters from the endpoint definition.
        Returns:
        list: List of required parameters for the given endpoint.
        """
        required_params = []
        parameters = self.tool_config.get('parameter', {}).get('properties', {})

        # Check each parameter to see if it is required
        for param, details in parameters.items():
            if details.get('required', False):
                required_params.append(param.lower())

        return required_params