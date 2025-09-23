import yaml
import json


def yaml_to_dict(yaml_file_path):
    """
    Convert a YAML file to a dictionary.

    Args:
        yaml_file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary representation of the YAML file content.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            yaml_dict = yaml.safe_load(file)
            return yaml_dict
    except FileNotFoundError:
        print(f"File not found: {yaml_file_path}")
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}")


def read_json_list(file_path):
    """
    Reads a list of JSON objects from a file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    list: A list of dictionaries containing the JSON objects.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def evaluate_function_call(tool_definition, function_call):
    # Map for type conversion
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }

    # Check if the function name matches
    if tool_definition["name"] != function_call["name"]:
        return False, "Function name does not match."

    # Check if all required parameters are present
    required_params = [key for key, value in tool_definition["parameter"]
                       ["properties"].items() if value.get("required", False)]
    missing_params = [
        param for param in required_params if param not in function_call["arguments"]]
    if missing_params:
        return False, f"Missing required parameters: {missing_params}"

    # Check if all provided parameters are valid and their data types are correct
    valid_params = tool_definition["parameter"]["properties"]
    invalid_params = []
    type_mismatches = []

    for param, value in function_call["arguments"].items():
        if param not in valid_params:
            invalid_params.append(param)
        else:
            expected_type = valid_params[param]["type"]
            if expected_type not in type_map:
                return False, f"Unsupported parameter type: {expected_type}"
            if not isinstance(value, type_map[expected_type]):
                type_mismatches.append(
                    (param, expected_type, type(value).__name__))

    if invalid_params:
        return False, f"Invalid parameters provided: {invalid_params}"

    if type_mismatches:
        return False, f"Type mismatches: {type_mismatches}"

    return True, "Function call is valid."

def evaluate_function_call_from_toolbox(toolbox, function_call):
    tool_name = function_call["name"]
    this_tool_dec = toolbox.get_one_tool_by_one_name(tool_name)
    if this_tool_dec is None:
        return False, "Tool not found."
    results, results_message = evaluate_function_call(this_tool_dec, function_call)
    return results, results_message
    

def compare_function_calls(pred_function_call, gt_function_call, compare_arguments=True, compare_value=True):
    # Extracting the name and arguments from the predicted function call
    pred_name = pred_function_call["name"]
    pred_arguments = pred_function_call["arguments"]

    # Extracting the name and arguments from the ground truth function call
    gt_name = gt_function_call["name"]
    gt_arguments = gt_function_call["arguments"]

    # Compare function names
    if pred_name != gt_name:
        return False, "Function names do not match."

    if compare_arguments:
        # Compare arguments
        if set(pred_arguments.keys()) != set(gt_arguments.keys()):
            missing_in_pred = set(gt_arguments.keys()) - set(pred_arguments.keys())
            missing_in_gt = set(pred_arguments.keys()) - set(gt_arguments.keys())
            return False, f"Argument keys do not match. Missing in predicted: {missing_in_pred}, Missing in ground truth: {missing_in_gt}"
    if compare_value:
        # Compare argument values
        mismatched_values = []
        for key in pred_arguments:
            if pred_arguments[key] != gt_arguments[key]:
                mismatched_values.append((key, pred_arguments[key], gt_arguments[key]))

        if mismatched_values:
            return False, f"Argument values do not match: {mismatched_values}"

    return True, "Function calls match."


def extract_function_call_json(lst, return_message=False, verbose=True):
    if type(lst) is dict:
        if return_message:
            return lst, ""
        return lst
    result_str = ''.join(lst)
    if verbose:
        print("\033[1;34mPossible LLM outputs for function call:\033[0m", result_str)
    try:
        function_call_json = json.loads(result_str.strip())
        if return_message:
            return function_call_json, ""
        return function_call_json
    except json.JSONDecodeError:
        try:
            index_start = result_str.find(
                '[TOOL_CALLS]') 
            index_end = result_str.find('</s>')
            if index_end == -1:
                index_end = result_str.find('<|eom_id|>')
            if index_end == -1:
                function_call_str = result_str[index_start+ len('[TOOL_CALLS]'):]
            else:
                function_call_str = result_str[index_start+ len('[TOOL_CALLS]'):index_end]
            # print("function_call_str", function_call_str)
            function_call_json = json.loads(function_call_str.strip())
            if return_message:
                message = result_str[:index_start]
                return function_call_json, message
            return function_call_json
        except json.JSONDecodeError:
            try:
                print("Multiple function calls not implemented for 'llama' format.")
                index_start = result_str.find(
                    '<functioncall>') + len('<functioncall>')
                index_end = result_str.find('</functioncall>')
                function_call_str = result_str[index_start:index_end]
                # function_call_str = function_call_str.replace("'", '"')
                function_call_json = json.loads(function_call_str.strip())
                return function_call_json
            except json.JSONDecodeError as e:
                print("Not a function call:", e)
                if return_message:
                    return None, result_str
                return None
