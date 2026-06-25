from fastapi import FastAPI

app = FastAPI(title="Financial Helpdesk Agent")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
