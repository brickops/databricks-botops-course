# Databricks notebook source
# MAGIC %md 
# MAGIC # Mosaic AI Agent Framework: Author and deploy a tool-calling OpenAI Responses API agent
# MAGIC
# MAGIC This notebook demonstrates how to author an OpenAI agent that's compatible with Mosaic AI Agent Framework features. In this notebook you learn to:
# MAGIC - Author a tool-calling `ChatAgent` that uses the [Open AI Responses API ](https://platform.openai.com/docs/api-reference/responses)
# MAGIC - Manually test the agent's output
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC **Note**: This notebook queries the OpenAI REST API directly. For governance, payload logging, and other Databricks AI Gateway functionality, use Databricks external models ([AWS](https://docs.databricks.com/aws/en/generative-ai/external-models/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/external-models/)). For an example that uses Databricks external models to query OpenAI models, see the [OpenAI tool-calling agent notebook](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/openai-pyfunc-tool-calling-agent.html).
# MAGIC
# MAGIC To learn more about authoring an agent using Mosaic AI Agent Framework, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model)).
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Create a Databricks secret with your OpenAI API key ([AWS](https://docs.databricks.com/aws/en/security/secrets/example-secret-workflow) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/example-secret-workflow))
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow backoff databricks-openai openai databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC #### Agent tools
# MAGIC This agent code adds the built-in Unity Catalog function `system.ai.python_exec` to the agent. The agent code also includes commented-out sample code for adding a vector search index to perform unstructured data retrieval.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool))
# MAGIC

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC from typing import Any, Callable, Generator, Optional, Union
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC from openai import OpenAI
# MAGIC from openai.types.chat import ChatCompletionToolParam
# MAGIC from openai.types.responses import (
# MAGIC     ResponseFunctionToolCall,
# MAGIC     ResponseOutputItem,
# MAGIC     ResponseOutputMessage,
# MAGIC )
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your desired OpenAI model
# MAGIC # Databricks does not yet support the Responses API
# MAGIC LLM_ENDPOINT_NAME = "gpt-4o"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = """
# MAGIC You are a helpful assistant.
# MAGIC """
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC class ToolInfo(BaseModel):
# MAGIC     """
# MAGIC     Class representing a tool for the agent.
# MAGIC     - "name" (str): The name of the tool.
# MAGIC     - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
# MAGIC     - "exec_fn" (Callable): Function that implements the tool logic
# MAGIC     """
# MAGIC
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC
# MAGIC def convert_chat_completion_tool_to_tool_info(
# MAGIC     spec: ChatCompletionToolParam, exec_fn: Optional[Callable] = None
# MAGIC ):
# MAGIC     """Converts a ChatCompletionToolParam to a ToolInfo object."""
# MAGIC     spec |= spec.pop("function")
# MAGIC     if exec_fn is None:
# MAGIC
# MAGIC         def exec_fn(**kwargs):
# MAGIC             udf_name = spec["name"].replace("__", ".")
# MAGIC             function_result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC             if function_result.error is not None:
# MAGIC                 return function_result.error
# MAGIC             else:
# MAGIC                 return function_result.value
# MAGIC
# MAGIC     return ToolInfo(name=spec["name"], spec=spec, exec_fn=exec_fn)
# MAGIC
# MAGIC
# MAGIC TOOL_INFOS = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC # Below, we add the `system.ai.python_exec` UDF, which provides
# MAGIC # a python code interpreter tool to our agent
# MAGIC
# MAGIC # TODO: Add additional tools
# MAGIC UC_TOOL_NAMES = ["system.ai.python_exec"]
# MAGIC
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     TOOL_INFOS.append(convert_chat_completion_tool_to_tool_info(tool_spec))
# MAGIC
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# MAGIC # for details
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC
# MAGIC # TODO: Add vector search indexes
# MAGIC # VECTOR_SEARCH_TOOLS.append(
# MAGIC #     VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # )
# MAGIC for vs_tool in VECTOR_SEARCH_TOOLS:
# MAGIC     TOOL_INFOS.append(
# MAGIC         convert_chat_completion_tool_to_tool_info(vs_tool.tool, vs_tool.execute)
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC class ToolCallingAgent(ChatAgent):
# MAGIC     """
# MAGIC     Class representing a tool-calling Agent
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
# MAGIC         """
# MAGIC         Initializes the ToolCallingAgent with tools.
# MAGIC         """
# MAGIC         super().__init__()
# MAGIC         self.llm_endpoint = llm_endpoint
# MAGIC         self.client: OpenAI = OpenAI()
# MAGIC         self._tools_dict = {tool.name: tool for tool in tools}
# MAGIC
# MAGIC     def get_tool_specs(self) -> list[dict]:
# MAGIC         """
# MAGIC         Returns tool specifications in the format OpenAI expects.
# MAGIC         """
# MAGIC         return [tool_info.spec for tool_info in self._tools_dict.values()]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name: str, args: dict) -> Any:
# MAGIC         """
# MAGIC         Executes the specified tool with the given arguments.
# MAGIC         """
# MAGIC         if tool_name not in self._tools_dict:
# MAGIC             raise ValueError(f"Unknown tool: {tool_name}")
# MAGIC         return self._tools_dict[tool_name].exec_fn(**args)
# MAGIC
# MAGIC     def prepare_messages_for_llm(
# MAGIC         self,
# MAGIC         messages: list[Union[ChatAgentMessage, ResponseOutputItem, dict[str, Any]]],
# MAGIC     ) -> list[Union[dict[str, Any], ResponseOutputItem]]:
# MAGIC         """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
# MAGIC         compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
# MAGIC         return [
# MAGIC             {
# MAGIC                 k: v
# MAGIC                 for k, v in m.model_dump_compat(exclude_none=True).items()
# MAGIC                 if k in compatible_keys
# MAGIC             }
# MAGIC             if isinstance(m, ChatAgentMessage)
# MAGIC             else m
# MAGIC             for m in messages
# MAGIC         ]
# MAGIC
# MAGIC     def convert_openai_response_output_to_chat_agent_msg(
# MAGIC         self, output: ResponseOutputItem
# MAGIC     ) -> ChatAgentMessage:
# MAGIC         """Converts an OpenAI ResponseOutputItem to a ChatAgentMessage."""
# MAGIC         if isinstance(output, ResponseFunctionToolCall):
# MAGIC             return ChatAgentMessage(
# MAGIC                 **{
# MAGIC                     "role": "assistant",
# MAGIC                     "id": output.id,
# MAGIC                     "content": "",
# MAGIC                     "tool_calls": [
# MAGIC                         {
# MAGIC                             "id": output.call_id,
# MAGIC                             "type": "function",
# MAGIC                             "function": {
# MAGIC                                 "name": output.name,
# MAGIC                                 "arguments": output.arguments,
# MAGIC                             },
# MAGIC                         }
# MAGIC                     ],
# MAGIC                 }
# MAGIC             )
# MAGIC         elif isinstance(output, ResponseOutputMessage):
# MAGIC             return ChatAgentMessage(
# MAGIC                 role=output.role, content=output.content[0].text, id=output.id
# MAGIC             )
# MAGIC         else:
# MAGIC             raise NotImplementedError("Add more parsing for other output types")
# MAGIC
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def chat_completion(
# MAGIC         self, messages: list[Union[ChatAgentMessage, ResponseOutputItem, dict]]
# MAGIC     ) -> ResponseOutputItem:
# MAGIC         return self.client.responses.create(
# MAGIC             model=self.llm_endpoint,
# MAGIC             input=self.prepare_messages_for_llm(messages),
# MAGIC             tools=self.get_tool_specs(),
# MAGIC         ).output[0]
# MAGIC
# MAGIC     def handle_tool_call(
# MAGIC         self,
# MAGIC         llm_output: ResponseFunctionToolCall,
# MAGIC         current_msg_history: list[
# MAGIC             Union[ChatAgentMessage, ResponseOutputItem, dict[str, Any]]
# MAGIC         ],
# MAGIC     ) -> ChatAgentMessage:
# MAGIC         """
# MAGIC         Execute tool calls, add them to the running message history, and return a tool ChatAgentMessage
# MAGIC         """
# MAGIC         args = json.loads(llm_output.arguments)
# MAGIC         result = str(self.execute_tool(tool_name=llm_output.name, args=args))
# MAGIC
# MAGIC         # format from step 4 https://platform.openai.com/docs/guides/function-calling#function-calling-steps
# MAGIC         openai_response_tool_msg = {
# MAGIC             "type": "function_call_output",
# MAGIC             "call_id": llm_output.call_id,
# MAGIC             "output": result,
# MAGIC         }
# MAGIC         current_msg_history.append(openai_response_tool_msg)
# MAGIC         return ChatAgentMessage(
# MAGIC             role="tool",
# MAGIC             name=llm_output.name,
# MAGIC             tool_call_id=llm_output.call_id,
# MAGIC             content=result,
# MAGIC             id=str(uuid4()),
# MAGIC         )
# MAGIC
# MAGIC     def call_and_run_tools(
# MAGIC         self,
# MAGIC         messages: list[Union[ChatAgentMessage, ResponseOutputItem, dict[str, Any]]],
# MAGIC         max_iter: int = 10,
# MAGIC     ) -> Generator[ChatAgentMessage, None, None]:
# MAGIC         for i in range(max_iter):
# MAGIC             llm_output = self.chat_completion(messages=messages)
# MAGIC             messages.append(llm_output)
# MAGIC             yield self.convert_openai_response_output_to_chat_agent_msg(llm_output)
# MAGIC
# MAGIC             if not isinstance(llm_output, ResponseFunctionToolCall):
# MAGIC                 return  # Stop streaming if no tool calls are needed
# MAGIC
# MAGIC             yield self.handle_tool_call(llm_output, messages)
# MAGIC
# MAGIC         yield ChatAgentMessage(
# MAGIC             content=f"I'm sorry, I couldn't determine the answer after trying {max_iter} times.",
# MAGIC             role="assistant",
# MAGIC             id=str(uuid4()),
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         """
# MAGIC         Primary function that takes a user's request and generates a response.
# MAGIC         """
# MAGIC         # NOTE: this assumes that each chunk streamed by self.call_and_run_tools contains
# MAGIC         # a full message (i.e. chunk.delta is a complete message).
# MAGIC         # This is simple to implement, but you can also stream partial response messages from predict_stream,
# MAGIC         # and aggregate them in predict_stream by message ID
# MAGIC         response_messages = [
# MAGIC             chunk.delta
# MAGIC             for chunk in self.predict_stream(messages, context, custom_inputs)
# MAGIC         ]
# MAGIC         return ChatAgentResponse(messages=response_messages)
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         if len(messages) == 0:
# MAGIC             raise ValueError("`messages` must contain at least one message")
# MAGIC         all_messages = [
# MAGIC             ChatAgentMessage(role="system", content=SYSTEM_PROMPT)
# MAGIC         ] + messages
# MAGIC
# MAGIC         for message in self.call_and_run_tools(messages=all_messages):
# MAGIC             yield ChatAgentChunk(delta=message)
# MAGIC
# MAGIC # Log the model using MLflow
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since we manually traced methods within `ChatAgent`, you can view the trace for each step the agent takes, with any LLM calls made via the OpenAI SDK automatically traced by autologging.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

# TODO: set secret_scope_name and secret_key_name to access your OpenAI API key
secret_scope_name = ""
secret_key_name = ""
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    scope=secret_scope_name, key=secret_key_name
)
assert os.environ["OPENAI_API_KEY"] is not None, ("The OPENAI_API_KEY env var was not properly set")

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "What is 4*3 in Python?"}]})

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What is 4*3 in python?"}]}
):
    print(chunk, "-----------\n")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`
# MAGIC
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).
# MAGIC
# MAGIC

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction

resources = []
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        pip_requirements=[
            "mlflow",
            "backoff",
            "databricks-openai",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with Agent Evaluation
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evalaute the agent's responses based on expected responses and other evaluation criteria. Use the evaluation criteria you specify to guide iterations, using MLflow to track the computed quality metrics.
# MAGIC See Databricks documentation ([AWS]((https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)).
# MAGIC
# MAGIC
# MAGIC To evaluate your tool calls, add custom metrics. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls)).

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {"messages": [{"role": "user", "content": "What is an LLM agent?"}]},
        "expected_response": None,
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)

# COMMAND ----------

import mlflow

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = ""
schema = ""
model_name = ""
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "docs"},
    environment_vars={
        "OPENAI_API_KEY": f"{{{{secrets/{secret_scope_name}/{secret_key_name}}}}}"
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details
