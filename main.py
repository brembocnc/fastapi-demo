from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "恭喜！你的 API 已經成功部署到雲端了！"}

@app.get("/add")
def add_numbers(a: int, b: int):
    result = a + b
    return {
        "operation": "addition",
        "a": a,
        "b": b,
        "result": result
    }

@app.get("/hello/{name}")
def say_hello(name: str):
    return {"message": f"你好, {name}！來自雲端的問候。"}