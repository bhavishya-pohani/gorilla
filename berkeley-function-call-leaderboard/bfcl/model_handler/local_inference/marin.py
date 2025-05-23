import inspect
import json

from bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl.model_handler.utils import (
    func_doc_language_specific_pre_processing,
)
from overrides import override

class MarinHandler(OSSHandler):
    # copied from HermesHandler, is a prompting model
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    @override
    def _format_prompt(self, messages, function):
        tool_call_format = """{"arguments": <args-dict>, "name": <function-name>}"""
        formatted_prompt = inspect.cleandoc(
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
            You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:
            <tools>
            {function}
            </tools>
            For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
            <tool_call>
            {tool_call_format}
            </tool_call>
            <|eot_id|>
            """
        )

        formatted_prompt = formatted_prompt.format(
            function=function,
            tool_call_format=tool_call_format,
        )

        for message in messages:
            formatted_prompt += (
                f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>"
            )

        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

        return formatted_prompt

    @override
    def decode_ast(self, result, language="Python"):
        lines = result.split("\n")
        flag = False
        func_call = []
        for line in lines:
            if "<tool_call>" == line:
                flag = True
            elif "</tool_call>" == line:
                flag = False
            else:
                if flag:
                    line = line.replace("'", '"')
                    tool_result = json.loads(line)
                    func_call.append({tool_result["name"]: tool_result["arguments"]})
                flag = False
        return func_call

    @override
    def decode_execute(self, result):
        lines = result.split("\n")
        flag = False
        function_call_list = []
        for line in lines:
            if "<tool_call>" == line:
                flag = True
            elif "</tool_call>" == line:
                flag = False
            else:
                if flag:
                    line = line.replace("'", '"')
                    tool_result = json.loads(line)
                    function_call_list.append(
                        {tool_result["name"]: tool_result["arguments"]}
                    )
                flag = False
        execution_list = []
        for function_call in function_call_list:
            for key, value in function_call.items():
                execution_list.append(
                    f"{key}({','.join([f'{k}={repr(v)}' for k,v in value.items()])})"
                )
        return execution_list

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        
        functions = func_doc_language_specific_pre_processing(functions, test_category)

        # Marin uses its own system prompt

        return {"message": [], "function": functions}

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            marin_response_object = {
                "name": decoded_model_response,
                "content": execution_result,
            }
            inference_data["message"].append(
                {
                    "role": "tool",
                    "content": f"<tool_response>\n{marin_response_object}\n</tool_response>\n",
                }
            )

        return inference_data
