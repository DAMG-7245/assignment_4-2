# rag-pipeline/api/main.py
from fastapi import FastAPI
from routes.routes import router  

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Pipeline API")
    app.include_router(router, prefix="/api")
    return app

app = create_app()

# 如果要直接运行: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
