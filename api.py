from typing import Any
from informationClass import InformationProcessor
from langchain.prompts.prompt import PromptTemplate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # 我們使用 Pydantic 來使用標準的 Python 型別聲明請求。
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# app = FastAPI(docs_url=None, redoc_url=None)  # 关闭 Swagger UI 和 ReDoc UI
# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源。
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InformationRequest(BaseModel):
    information: str


@app.post("/process_information")
async def process_information(request: InformationRequest) -> Any:
    try:
        processor = InformationProcessor(
            information=request.information, model_name="gpt-3.5-turbo", temperature=0)
        prompt_template = processor.create_prompt_template()
        processor.create_llm_chain(prompt_template)
        result = processor.process_information()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
