from .graphql_tool import GraphQLTool
import requests
import copy
import json

def execute_RESTful_query(endpoint_url, variables=None):
    response = requests.get(
        endpoint_url, params = variables)
    try:
        result = response.json()

        # Check if the response contains errors
        if 'error' in result:
            print("Invalid Query: ", result['error'])
            return False
        else:
            return result
    except requests.exceptions.JSONDecodeError as e:
        print("JSONDecodeError: Could not decode the response as JSON")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

class RESTfulTool(GraphQLTool):
    def __init__(self, tool_config, endpoint_url):
        super().__init__(tool_config, endpoint_url)

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        return execute_RESTful_query(endpoint_url=self.endpoint_url, variables=arguments)

class MonarchTool(RESTfulTool):
    def __init__(self, tool_config):
        endpoint_url = 'https://api.monarchinitiative.org/v3/api' + tool_config['tool_url']
        super().__init__(tool_config, endpoint_url)
    
    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        query_schema_runtime = copy.deepcopy(self.query_schema)
        for key in query_schema_runtime:
            if key in arguments:
                query_schema_runtime[key] = arguments[key]
        if "url_key" in query_schema_runtime:
            url_key_name = query_schema_runtime['url_key']
            formatted_endpoint_url = self.endpoint_url.format(url_key = query_schema_runtime[url_key_name])
            del query_schema_runtime['url_key']
        else:
            formatted_endpoint_url = self.endpoint_url
        if isinstance(query_schema_runtime, dict):
            print(query_schema_runtime)
            if 'query' in query_schema_runtime:
                query_schema_runtime['q'] = query_schema_runtime['query'] # match with the api
        response = execute_RESTful_query(endpoint_url=formatted_endpoint_url, variables=query_schema_runtime)
        if 'facet_fields' in response:
            del response['facet_fields']
        def remove_empty_values(obj):
            if isinstance(obj, dict):
                return {k: remove_empty_values(v) for k, v in obj.items()
                        if v not in [0, [], None]}
            elif isinstance(obj, list):
                return [remove_empty_values(v) for v in obj if v not in [0, [], None]]
            else:
                return obj
        response = remove_empty_values(response)
        return response

class MonarchDiseasesForMultiplePhenoTool(MonarchTool):
    def __init__(self, tool_config):
        super().__init__(tool_config)
    
    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        query_schema_runtime = copy.deepcopy(self.query_schema)
        for key in query_schema_runtime:
            if (key!="HPO_ID_list") and (key in arguments):
                query_schema_runtime[key] = arguments[key]
        all_diseases = []
        for HPOID in arguments['HPO_ID_list']:
            each_query_schema_runtime = copy.deepcopy(query_schema_runtime)
            each_query_schema_runtime['object'] = HPOID
            each_query_schema_runtime['limit'] = 500
            each_output = execute_RESTful_query(endpoint_url=self.endpoint_url, variables=each_query_schema_runtime)
            each_output = each_output['items']
            each_output_names = [disease['subject_label'] for disease in each_output]
            all_diseases.append(each_output_names)
        
        intersection = set(all_diseases[0])
        for element in all_diseases[1:]:
            intersection &= set(element)
        intersection = list(intersection)
        if query_schema_runtime['limit'] < len(intersection):
            intersection = intersection[:query_schema_runtime['limit']]
        return intersection
