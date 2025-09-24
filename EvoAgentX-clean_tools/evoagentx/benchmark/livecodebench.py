import os 
# import regex
from typing import Union, Any, List
from ..core.logging import logger
from .benchmark import CodingBenchmark 
from ..core.module_utils import extract_code_blocks
from .lcb_utils.code_generation import (
    CodeGenerationProblem, 
    load_code_generation_dataset
)
from .lcb_utils.test_output_prediction import (
    TestOutputPredictionProblem, 
    load_test_prediction_dataset
)
from .lcb_utils.code_execution import (
    CodeExecutionProblem, 
    load_code_execution_dataset
)
from .lcb_utils.evaluation import (
    codegen_metrics, 
    test_output_metrics,
    code_execution_metrics
)
from .lcb_utils.utils import extract_test_output_code, extract_execution_code


VALID_SCENARIO = ["code_generation", "test_output_prediction", "code_execution"]

class LiveCodeBench(CodingBenchmark):

    """Benchmark class for evaluating LLM capabilities on real-world programming tasks.
    
    LiveCodeBench provides a framework for evaluating different scenarios of code-related tasks:
    1. Code Generation: generating code from problem descriptions
    2. Test Output Prediction: predicting test outputs given test code
    3. Code Execution: generating code that executes correctly
    
    The benchmark supports different evaluation modes, metrics, and can be customized
    with various parameters like timeouts, sample dates, and processing options.
    
    Attributes:
        k: An integer or list of integers specifying which pass@k metrics to compute
        version: Release version of the dataset to use
        num_process: Number of processes to use for evaluation
        start_date: Filter problems to those after this date
        end_date: Filter problems to those before this date
        scenario: Type of programming task to evaluate ("code_generation", 
                  "test_output_prediction", or "code_execution")
        use_cot_for_execution: Whether to use chain-of-thought processing for code execution
    """

    def __init__(
        self, 
        path: str = None, 
        mode: str = "all", 
        timeout: int = 60, 
        k: Union[int, list] = 1, 
        num_process: int = 6, 
        scenario: str = "code_generation", 
        version: str = "release_latest", 
        start_date: str = None, 
        end_date: str = None, 
        use_cot_for_execution: bool = False, 
        **kwargs
    ):
        path = os.path.expanduser(path or "~/.evoagentx/data/livecodebench")
        self.k = k 
        self.version = version
        self.num_process = num_process
        self.start_date = start_date
        self.end_date = end_date
        self.scenario = scenario 
        self.use_cot_for_execution = use_cot_for_execution
        assert scenario in VALID_SCENARIO, f"Invalid scenario: {scenario}. Available choices: {VALID_SCENARIO}." 
        super().__init__(name=type(self).__name__, path=path, mode=mode, timeout=timeout, **kwargs)
    
    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = None 
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = None 
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_test_data()
    
    def _load_test_data(self):

        if self.scenario == "code_generation":
            logger.info(f"Loading code generation dataset from {self.path} with version {self.version}.")
            data: List[CodeGenerationProblem] = load_code_generation_dataset(
                release_version=self.version, 
                cache_dir=self.path, 
                start_date=self.start_date, 
                end_date=self.end_date
            )
        elif self.scenario == "test_output_prediction":
            logger.info(f"Loading test output prediction dataset from {self.path}.")
            data: List[TestOutputPredictionProblem] = load_test_prediction_dataset(cache_dir=self.path)
        elif self.scenario == "code_execution":
            logger.info(f"Loading code execution dataset from {self.path}.")
            data: List[CodeExecutionProblem] = load_code_execution_dataset(cache_dir=self.path)
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}. Available choices: {VALID_SCENARIO}.")

        return data 
    
    def _get_id(self, example: Union[CodeGenerationProblem, TestOutputPredictionProblem]) -> str:
        return example.question_id  
    
    def _get_label(self, example: Union[CodeGenerationProblem, TestOutputPredictionProblem]) -> dict:
        return example.get_evaluation_sample()
    
    def evaluate(self, prediction: Any, label: Any) -> dict:
        """
        Evaluate the solution code.

        Args:
            prediction (str | List[str]): The solution code(s).
            label (dict | List[dict]): The test cases and expected outputs. 

        Returns:
            dict: The evaluation metrics (pass@k).
        """
        prediction, label = self._check_evaluation_inputs(prediction, label)
        k_list = [self.k] if isinstance(self.k, int) else self.k

        if self.scenario == "code_generation":
            solutions: List[str] = [extract_code_blocks(pred)[0] for pred in prediction]
            metrics, results, metadatas = codegen_metrics(
                samples_list=label, # label is already a list 
                generations_list=[solutions], # for a single example. 
                k_list=k_list, 
                num_process_evaluate=self.num_process,
                timeout=self.timeout
            )
            
        elif self.scenario == "test_output_prediction":
            pred_outputs = [extract_test_output_code(pred) for pred in prediction]
            metrics, results = test_output_metrics(
                samples=label, 
                generations=[pred_outputs], 
                k_list=k_list, 
            )
        elif self.scenario == "code_execution":
            pred_outputs = [extract_execution_code(pred, self.use_cot_for_execution) for pred in prediction]
            metrics, results = code_execution_metrics(
                samples=label, 
                generations=[pred_outputs], 
            )
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}. Available choices: {VALID_SCENARIO}.")
        
        pass_at_k = {f"pass@{k}": float(metrics[f"pass@{k}"]) for k in k_list}
        return pass_at_k
