from graphql import build_schema
from graphql.language import parse
from graphql.validation import validate
from .base_tool import BaseTool
import requests
import copy
import json


def validate_query(query_str, schema_str):
    try:
        # Build the GraphQL schema object from the provided schema string
        schema = build_schema(schema_str)

        # Parse the query string into an AST (Abstract Syntax Tree)
        query_ast = parse(query_str)

        # Validate the query AST against the schema
        validation_errors = validate(schema, query_ast)

        if not validation_errors:
            return True
        else:
            # Collect and return the validation errors
            error_messages = '\n'.join(str(error)
                                       for error in validation_errors)
            return f"Query validation errors:\n{error_messages}"
    except Exception as e:
        return f"An error occurred during validation: {str(e)}"

def remove_none_and_empty_values(json_obj):
    """Remove all key-value pairs where the value is None or an empty list"""
    if isinstance(json_obj, dict):
        return {k: remove_none_and_empty_values(v) for k, v in json_obj.items() if v is not None and v != []}
    elif isinstance(json_obj, list):
        return [remove_none_and_empty_values(item) for item in json_obj if item is not None and item != []]
    else:
        return json_obj

def execute_query(endpoint_url, query, variables=None):
    response = requests.post(
        endpoint_url, json={'query': query, 'variables': variables})
    try:
        result = response.json()
        # result = json.dumps(result, ensure_ascii=False)
        result = remove_none_and_empty_values(result)
        # Check if the response contains errors
        if 'errors' in result:
            print("Invalid Query: ", result['errors'])
            return None
         # Check if the data field is empty
        elif not result.get('data') or all(not v for v in result['data'].values()):
            print("No data returned")
            return None
        else:
            return result
    except requests.exceptions.JSONDecodeError as e:
        print("JSONDecodeError: Could not decode the response as JSON")
        return None


class GraphQLTool(BaseTool):
    def __init__(self, tool_config, endpoint_url):
        super().__init__(tool_config)
        self.endpoint_url = endpoint_url
        self.query_schema = tool_config['query_schema']
        self.parameters = tool_config['parameter']['properties']
        self.default_size = 5

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        if 'size' in self.parameters and 'size' not in arguments:
            arguments['size'] = 5
        return execute_query(endpoint_url=self.endpoint_url, query=self.query_schema, variables=arguments)


class OpentargetTool(GraphQLTool):
    def __init__(self, tool_config):
        endpoint_url = 'https://api.platform.opentargets.org/api/v4/graphql'
        super().__init__(tool_config, endpoint_url)
        
    def run(self, arguments):
        for each_arg, arg_value in arguments.items(): # opentarget api cannot handle '-' in the arguments
            if isinstance(arg_value, str): 
                if '-' in arg_value:
                    arguments[each_arg] = arg_value.replace('-', ' ')
        return super().run(arguments)
            

class OpentargetToolDrugNameMatch(GraphQLTool):
    def __init__(self, tool_config, drug_generic_tool=None):
        endpoint_url = 'https://api.platform.opentargets.org/api/v4/graphql'
        self.drug_generic_tool = drug_generic_tool
        self.possible_drug_name_args = ['drugName']
        super().__init__(tool_config, endpoint_url)

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        results = execute_query(endpoint_url=self.endpoint_url, query=self.query_schema, variables=arguments)
        if results is None:
            print("No results found for the drug brand name. Trying with the generic name.")
            name_arguments = {}
            for each_args in self.possible_drug_name_args:
                if each_args in arguments:
                    name_arguments['drug_name'] = arguments[each_args]
                    break
            if len(name_arguments)==0:
                print("No drug name found in the arguments.")
                return None
            drug_name_results = self.drug_generic_tool.run(name_arguments)
            if drug_name_results is not None and 'openfda.generic_name' in drug_name_results:
                arguments[each_args] = drug_name_results['openfda.generic_name']
                print("Found generic name. Trying with the generic name: ", arguments[each_args])
                results = execute_query(endpoint_url=self.endpoint_url, query=self.query_schema, variables=arguments)
        return results
            


class OpentargetGeneticsTool(GraphQLTool):
    def __init__(self, tool_config):
        endpoint_url = 'https://api.genetics.opentargets.org/graphql'
        super().__init__(tool_config, endpoint_url)
