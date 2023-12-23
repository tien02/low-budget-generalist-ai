import config
import uvicorn
import asyncio
from langchain.llms import LlamaCpp

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

TEMPLATE = """You are the AI assisstant which specialized in Medical domain. 
Your response should be in plain text and short with supportive information.
Let's think step by step.
Be concise.

Question: {question}

Answer: """

llm = LlamaCpp(
    model_path=config.LLM_CKPT,
    temperature=0.7,
    max_tokens=5000,
    top_p=0.95,
    verbose=False,
    streaming=True,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def generate_response(question:str):
    try: 
        prompt = TEMPLATE.format(question=question)

        for chunk in llm.stream(prompt, stop=["Question:"]):
            yield chunk
            await asyncio.sleep(0.5)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
def home():
    return {
        "message": "This is the quantized version of LLaMa2 7B specific for Medical Question Answering"
    }
    

@app.post('/stream_tokens')
def streaming_generation(
    question:str= Query(...,description="User's question",)
):
    return StreamingResponse(generate_response(question), media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run("llm:app", host="0.0.0.0", port=config.LLM_PORT)