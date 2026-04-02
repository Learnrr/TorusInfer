from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from python.openai_obj import(
    ChatCompletion,
    ChatCompletionResponse,
    Completion,
    CompletionResponse
)
from python.adaptor import(
    chat_completion_to_request,
    completion_to_request,
    request_to_chat_completion_response,
    request_to_completion_response
)
from python.engine import Engine

app = FastAPI()
config_path = "/llm_infer_engine/llm_engine_config.json"
engine = Engine(config_path)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletion) -> ChatCompletionResponse:
    api_request_obj = chat_completion_to_request(request)
    request_id = await run_in_threadpool(engine.submit, api_request_obj.prompt, api_request_obj)
    output = await run_in_threadpool(engine.get_output, request_id)
    response = request_to_chat_completion_response(api_request_obj, output)
    return response




@app.post("/v1/completions")
async def create_completion(request: Completion) -> CompletionResponse:
    api_request_obj = completion_to_request(request)
    request_id = await run_in_threadpool(engine.submit, api_request_obj.prompt, api_request_obj)
    output = await run_in_threadpool(engine.get_output, request_id)
    response = request_to_completion_response(api_request_obj, output)
    return response