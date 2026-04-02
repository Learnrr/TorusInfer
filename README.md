# llm_infer_engine
```
git clone ...

cd llm_infer_engine/weights

git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

cd llm_infer_engine

make all
// start serving
python3 -m uvicorn python.fastapi:app --host 0.0.0.0 --port 8000 --workers 1
//curl through api
curl -s http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    "model":"qwen",
    "messages":[
      {"role":"user","content":"Hello, please introduce yourself briefly."}
    ],
    "max_tokens":128,
    "temperature":0.7
  }'
```

