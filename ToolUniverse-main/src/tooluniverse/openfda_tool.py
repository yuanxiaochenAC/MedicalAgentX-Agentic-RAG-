import requests
from .base_tool import BaseTool
import copy
import re
import json
import os

def check_keys_present(api_capabilities_dict, keys):
    for key in keys:
        levels = key.split('.')
        current_dict = api_capabilities_dict
        key_present = True
        for level in levels:
            if level not in current_dict:
                print(f"Key '{level}' not found in dictionary.")
                key_present = False
                break
            if 'properties' in current_dict[level]:
                current_dict = current_dict[level]['properties']
            else:
                current_dict = current_dict[level]
    return key_present


def extract_nested_fields(records, fields, keywords=None):
    """
    Recursively extracts nested fields from a list of dictionaries.

    :param records: List of dictionaries from which to extract fields
    :param fields: List of nested fields to extract, each specified with dot notation (e.g., 'openfda.brand_name')

    :return: List of dictionaries containing only the specified fields
    """
    extracted_records = []
    for record in records:
        extracted_record = {}
        for field in fields:
            keys = field.split('.')
            # print("keys", keys)
            value = record
            try:
                for key in keys:
                    value = value[key]
                if key != 'openfda' and key !='generic_name' and key !='brand_name':
                    if len(keywords)>0:
                        # print("key words:", keywords)
                        # print(value)
                        # print(type(value))
                        value = extract_sentences_with_keywords(value, keywords)
                extracted_record[field] = value
            except KeyError:
                extracted_record[field] = None
        if any(extracted_record.values()):
            extracted_records.append(extracted_record)
    return extracted_records

def map_properties_to_openfda_fields(arguments, search_fields):
    """
    Maps the provided arguments to the corresponding openFDA fields based on the search_fields mapping.

    :param arguments: The input arguments containing property names and values.
    :param search_fields: The mapping of property names to openFDA fields.

    :return: A dictionary with openFDA fields and corresponding values.
    """
    mapped_arguments = {}

    for key, value in list(arguments.items()):
        if key in search_fields:
            # print("key in search_fields:", key)
            openfda_fields = search_fields[key]
            if isinstance(openfda_fields, list):
                for field in openfda_fields:
                    mapped_arguments[field] = value
            else:
                mapped_arguments[openfda_fields] = value
            del arguments[key]
    arguments['search_fields'] = mapped_arguments
    return arguments

def extract_sentences_with_keywords(text_list, keywords):
    """
    Extracts sentences containing any of the specified keywords from the text.
    
    Parameters:
    - text (str): The input text from which to extract sentences.
    - keywords (list): A list of keywords to search for in the text.
    
    Returns:
    - list: A list of sentences containing any of the keywords.
    """
    sentences_with_keywords = []
    for text in text_list:
        # Compile a regular expression pattern for sentence splitting
        sentence_pattern = re.compile(r'(?<=[.!?]) +')
        # Split the text into sentences
        sentences = sentence_pattern.split(text)
        # Initialize a list to hold sentences with keywords
        
        
        # Iterate through each sentence
        for sentence in sentences:
            # Check if any of the keywords are present in the sentence
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                # If a keyword is found, add the sentence to the list
                sentences_with_keywords.append(sentence)
    
    return "......".join(sentences_with_keywords)


def search_openfda(params=None, endpoint_url=None, api_key=None, sort=None, limit=5, 
                   skip=None, count=None, exists=None, return_fields=None, 
                   exist_option='OR', search_keyword_option='AND', keywords_filter=True):
    # Initialize params if not provided
    if params is None:
        params = {}
    
    if return_fields=='ALL':
        exists = None

    # Initialize search fields and construct search query
    search_fields = params.get('search_fields', {})
    search_query = []
    keywords_list = []
    if search_fields:
        for field, value in search_fields.items():
            # Merge multiple continuous black spaces into one and use one '+'
            if keywords_filter and field !='openfda.brand_name' and field !='openfda.generic_name':
                keywords_list.extend(value.split())
            if field == 'openfda.generic_name':
                value = value.upper() # all generic names are in uppercase
            value = value.replace(" and ", " ") # remove 'and' in the search query
            value = value.replace(" AND ", " ") # remove 'AND' in the search query
            value = ' '.join(value.split())
            if search_keyword_option=='AND':
                search_query.append(f'{field}:({value.replace(" ", "+AND+")})')
            elif search_keyword_option=='OR':
                search_query.append(f'{field}:({value.replace(" ", "+")})')
            else:
                print("Invalid search_keyword_option. Please use 'AND' or 'OR'.")
        del params['search_fields']
    if search_query:
        params['search'] = "+".join(search_query)
        params['search'] = '(' + params['search'] + ')'
    # Validate the presence of at least one of search, count, or sort
    if not (params.get('search') or params.get('count') or params.get('sort') or search_fields):
        return {"error": "You must provide at least one of 'search', 'count', or 'sort' parameters."}

    # Set additional query parameters
    params['limit'] = params.get('limit', limit)
    params['sort'] = params.get('sort', sort)
    params['skip'] = params.get('skip', skip)
    params['count'] = params.get('count', count)
    if exists is not None:
        if isinstance(exists, str):
            exists = [exists]
        if 'search' in params:
            if exist_option == 'AND':
                params['search'] += "+AND+(" + "+AND+".join(
                    [f"_exists_:{keyword}" for keyword in exists])+")"
            elif exist_option == 'OR':
                params['search'] += "+AND+(" + "+".join(
                    [f"_exists_:{keyword}" for keyword in exists])+")"
        else:
            if exist_option == 'AND':
                params['search'] = "+AND+".join(
                    [f"_exists_:{keyword}" for keyword in exists])
            elif exist_option == 'OR':
                params['search'] = "+".join(
                    [f"_exists_:{keyword}" for keyword in exists])
        # Ensure that at least one of the search fields exists
        params['search'] += "+AND+(" + "+".join(
            [f"_exists_:{field}" for field in search_fields.keys()]) +")"
        # params['search']+="+AND+_exists_:openfda"

    # Construct full query with additional parameters
    query = "&".join(
        [f"{key}={value}" for key, value in params.items() if value is not None])
    full_url = f"{endpoint_url}?{query}"
    if api_key:
        full_url += f"&api_key={api_key}"

    print(full_url)

    response = requests.get(full_url)

    # Get the JSON response
    response_data = response.json()
    if 'error' in response_data:
        print("Invalid Query: ", response_data['error'])
        return None

    # Extract meta information
    meta_info = response_data.get('meta', {})
    meta_info = meta_info.get('results', {})

    # Extract results and return only the specified return fields
    results = response_data.get('results', [])
    if return_fields == 'ALL':
        return {
            'meta': meta_info,
            'results': results
        }
    required_fields = list(search_fields.keys()) + return_fields
    extracted_results = extract_nested_fields(
        results, required_fields, keywords_list)
    return {
        'meta': meta_info,
        'results': extracted_results
    }


class FDATool(BaseTool):
    def __init__(self, tool_config, endpoint_url, api_key=None):
        super().__init__(tool_config)
        fields = tool_config['fields']
        self.search_fields = fields.get('search_fields', {})
        self.return_fields = fields.get('return_fields', [])
        self.exists = fields.get('exists', None)
        if self.exists is None:
            self.exists = self.return_fields
        self.endpoint_url = endpoint_url
        self.api_key = api_key or os.getenv('FDA_API_KEY')

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        mapped_arguments = map_properties_to_openfda_fields(
            arguments, self.search_fields)
        return search_openfda(mapped_arguments,
                              endpoint_url=self.endpoint_url,
                              api_key=self.api_key,
                              exists=self.exists,
                              return_fields=self.return_fields, exist_option='OR')


class FDADrugLabelTool(FDATool):
    def __init__(self, tool_config, api_key=None):
        endpoint_url = 'https://api.fda.gov/drug/label.json'
        super().__init__(tool_config, endpoint_url, api_key)


class FDADrugLabelSearchTool(FDATool):
    def __init__(self, tool_config=None, api_key=None):
        self.tool_config = {
            "name": "FDADrugLabelSearch",
            "description": "Retrieve information of a specific drug.",
            "label": ["search", "drug"],
            "type": "FDADrugLabelSearch",
            "parameter": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The name of the drug.",
                        "required": True
                    },
                    "return_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ['ALL', 'abuse', 'accessories', 'active_ingredient', 'adverse_reactions', 'alarms', 'animal_pharmacology_and_or_toxicology', 'ask_doctor', 'ask_doctor_or_pharmacist', 'assembly_or_installation_instructions', 'boxed_warning', 'calibration_instructions', 'carcinogenesis_and_mutagenesis_and_impairment_of_fertility', 'cleaning', 'clinical_pharmacology', 'clinical_studies', 'compatible_accessories', 'components', 'contraindications', 'controlled_substance', 'dependence', 'description', 'diagram_of_device', 'disposal_and_waste_handling', 'do_not_use', 'dosage_and_administration', 'dosage_forms_and_strengths', 'drug_abuse_and_dependence', 'drug_and_or_laboratory_test_interactions', 'drug_interactions', 'effective_time', 'environmental_warning', 'food_safety_warning', 'general_precautions', 'geriatric_use', 'guaranteed_analysis_of_feed', 'health_care_provider_letter', 'health_claim', 'how_supplied', 'id', 'inactive_ingredient', 'indications_and_usage', 'information_for_owners_or_caregivers', 'information_for_patients', 'instructions_for_use', 'intended_use_of_the_device', 'keep_out_of_reach_of_children', 'labor_and_delivery', 'laboratory_tests', 'mechanism_of_action', 'microbiology', 'nonclinical_toxicology', 'nonteratogenic_effects', 'nursing_mothers', 'openfda', 'other_safety_information', 'overdosage', 'package_label_principal_display_panel', 'patient_medication_information', 'pediatric_use', 'pharmacodynamics', 'pharmacogenomics', 'pharmacokinetics', 'precautions', 'pregnancy', 'pregnancy_or_breast_feeding', 'purpose', 'questions', 'recent_major_changes', 'references', 'residue_warning', 'risks', 'route', 'safe_handling_warning', 'set_id', 'spl_indexing_data_elements', 'spl_medguide', 'spl_patient_package_insert', 'spl_product_data_elements', 'spl_unclassified_section', 'statement_of_identity', 'stop_use', 'storage_and_handling', 'summary_of_safety_and_effectiveness', 'teratogenic_effects', 'troubleshooting', 'use_in_specific_populations', 'user_safety_warnings', 'version', 'warnings', 'warnings_and_cautions', 'when_using', 'meta'],
                            "description": "Searchable field."
                        },
                        "description": "Fields to search within drug labels.",
                        "required": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of records to return.",
                        "required": False
                    },
                    "skip": {
                        "type": "integer",
                        "description": "The number of records to skip.",
                        "required": False
                    }
                }
            },
            "fields": {
                "search_fields": {
                    "drug_name": ["openfda.brand_name", "openfda.generic_name"]
                },
            }
        }
        endpoint_url = 'https://api.fda.gov/drug/label.json'
        super().__init__(self.tool_config, endpoint_url, api_key)

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        mapped_arguments = map_properties_to_openfda_fields(
            arguments, self.search_fields)
        return_fields = arguments['return_fields']
        del arguments['return_fields']
        return search_openfda(mapped_arguments,
                              endpoint_url=self.endpoint_url,
                              api_key=self.api_key,
                              return_fields=return_fields, exists=return_fields, exist_option='OR')
        
class FDADrugLabelSearchIDTool(FDATool):
    def __init__(self, tool_config=None, api_key=None):
        self.tool_config = {
            "name": "FDADrugLabelSearchALLTool",
            "description": "Retrieve any related information to the query.",
            "label": ["search", "drug"],
            "type": "FDADrugLabelSearch",
            "parameter": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "key words need to be searched.",
                        "required": True
                    },
                    "return_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ['ALL', 'abuse', 'accessories', 'active_ingredient', 'adverse_reactions', 'alarms', 'animal_pharmacology_and_or_toxicology', 'ask_doctor', 'ask_doctor_or_pharmacist', 'assembly_or_installation_instructions', 'boxed_warning', 'calibration_instructions', 'carcinogenesis_and_mutagenesis_and_impairment_of_fertility', 'cleaning', 'clinical_pharmacology', 'clinical_studies', 'compatible_accessories', 'components', 'contraindications', 'controlled_substance', 'dependence', 'description', 'diagram_of_device', 'disposal_and_waste_handling', 'do_not_use', 'dosage_and_administration', 'dosage_forms_and_strengths', 'drug_abuse_and_dependence', 'drug_and_or_laboratory_test_interactions', 'drug_interactions', 'effective_time', 'environmental_warning', 'food_safety_warning', 'general_precautions', 'geriatric_use', 'guaranteed_analysis_of_feed', 'health_care_provider_letter', 'health_claim', 'how_supplied', 'id', 'inactive_ingredient', 'indications_and_usage', 'information_for_owners_or_caregivers', 'information_for_patients', 'instructions_for_use', 'intended_use_of_the_device', 'keep_out_of_reach_of_children', 'labor_and_delivery', 'laboratory_tests', 'mechanism_of_action', 'microbiology', 'nonclinical_toxicology', 'nonteratogenic_effects', 'nursing_mothers', 'openfda', 'other_safety_information', 'overdosage', 'package_label_principal_display_panel', 'patient_medication_information', 'pediatric_use', 'pharmacodynamics', 'pharmacogenomics', 'pharmacokinetics', 'precautions', 'pregnancy', 'pregnancy_or_breast_feeding', 'purpose', 'questions', 'recent_major_changes', 'references', 'residue_warning', 'risks', 'route', 'safe_handling_warning', 'set_id', 'spl_indexing_data_elements', 'spl_medguide', 'spl_patient_package_insert', 'spl_product_data_elements', 'spl_unclassified_section', 'statement_of_identity', 'stop_use', 'storage_and_handling', 'summary_of_safety_and_effectiveness', 'teratogenic_effects', 'troubleshooting', 'use_in_specific_populations', 'user_safety_warnings', 'version', 'warnings', 'warnings_and_cautions', 'when_using', 'meta'],
                            "description": "Searchable field."
                        },
                        "description": "Fields to search within drug labels.",
                        "required": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of records to return.",
                        "required": False
                    },
                    "skip": {
                        "type": "integer",
                        "description": "The number of records to skip.",
                        "required": False
                    }
                }
            },
            "fields": {
                "search_fields": {
                    "query": ['id']
                },
            }
        }
        endpoint_url = 'https://api.fda.gov/drug/label.json'
        super().__init__(self.tool_config, endpoint_url, api_key)

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)
        mapped_arguments = map_properties_to_openfda_fields(
            arguments, self.search_fields)
        return_fields = arguments['return_fields']
        del arguments['return_fields']
        return search_openfda(mapped_arguments,
                              endpoint_url=self.endpoint_url,
                              api_key=self.api_key,
                              return_fields=return_fields, exists=return_fields, exist_option='OR')


class FDADrugLabelGetDrugGenericNameTool(FDADrugLabelTool):
    def __init__(self, tool_config=None, api_key=None):
        
        if tool_config is None:
            tool_config = {
                "name": "get_drug_generic_name",
                "description": "Get the drug’s generic name based on the drug's generic or brand name.",
                "parameter": {
                    "type": "object",
                    "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "The generic or brand name of the drug.",
                        "required": True
                    }
                    }
                },
                "fields": {
                    "search_fields": {
                    "drug_name": [
                        "openfda.brand_name",
                        "openfda.generic_name"
                    ]
                    },
                    "return_fields": [
                    "openfda.generic_name"
                    ]
                },
                "type": "FDADrugLabelGetDrugGenericNameTool",
                "label": [
                    "FDADrugLabel",
                    "purpose",
                    "FDA"
                ]
                }
        

        from .data.fda_drugs_with_brand_generic_names_for_tool import drug_list
        
        self.brand_to_generic = {drug['brand_name']: drug['generic_name'] for drug in drug_list}
        self.generic_to_brand = {drug['generic_name']: drug['brand_name'] for drug in drug_list}
        
        super().__init__(tool_config, api_key)

    def run(self, arguments):
        
        drug_info = {}
        
        drug_name = arguments.get('drug_name')
        if '-' in drug_name:
            drug_name = drug_name.split('-')[0] # to handle some drug names such as tarlatamab-dlle
        if drug_name in self.brand_to_generic:
            drug_info['openfda.generic_name'] = self.brand_to_generic[drug_name]
            drug_info['openfda.brand_name'] = drug_name
        elif drug_name in self.generic_to_brand:
            drug_info['openfda.brand_name'] = self.generic_to_brand[drug_name]
            drug_info['openfda.generic_name'] = drug_name
        else:
            results = super().run(arguments)
            if results is not None:
                drug_info['openfda.generic_name'] = results['results'][0]['openfda.generic_name'][0]
                drug_info['openfda.brand_name'] = results['results'][0]['openfda.brand_name'][0]
                print("drug_info", drug_info)
            else:
                drug_info = None
        return drug_info  