# llm_infer_engine
supported model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
## quick start  
**1. clone to local**  
`git clone ...`  
**2. download weights**  
`cd llm_infer_engine/weights`  
`git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`  
**3. compile with make**  
`cd llm_infer_engine`  
`make clean`  
`make all`  
**4. start serving**  
`python3 -m uvicorn python.fastapi:app --host 0.0.0.0 --port 8000 --workers 1`   
**5. curl through api**  
```
curl -s http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    "model":"qwen",
    "messages":[
      {"role":"user","content":"Hello, please introduce yourself briefly."}
    ],
    "max_tokens":128,
    "temperature":0.7
  }'
```
**6. response like:**   
{"id":"chatcompletion-9d81e6046dd34b78946f14bc886e5e0d","object":"chat.completion","created":1775108401,"model":"qwen","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! I'm an AI assistant created by Anthropic to be helpful, harmless, and honest. How can I assist you today?","name":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":28,"total_tokens":43}}

**7. performance (needs improvement)**   
[2026-04-02 05:40:01] [INFO] /llm_infer_engine/src/Engine.cpp:124 - Sequence 1 metrics: Latency=4049ms, ITL=149ms, TPOT=149ms, TTFT=1035ms  
INFO:     127.0.0.1:43134 - "POST /v1/chat/completions HTTP/1.1" 200 OK  
