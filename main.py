from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from google import genai
from google.genai import types
import os
import tempfile
import json

app = FastAPI()

# --- 環境變數設定 (部署時在 Render 設定) ---
# 若 Render 沒設定，預設為你的程式碼中的值
PROJECT_ID = os.getenv("PROJECT_ID", "sudaocr")
LOCATION = os.getenv("GCP_LOCATION", "global") # 建議用 us-central1，global 有時會有延遲或權限問題
MODEL_ID = "gemini-3-pro-preview" # ⚠️ 注意：Gemini 3 Pro Preview 非常新，若無權限請改回 gemini-1.5-pro

# --- 初始化 Google GenAI Client ---
client = None

def setup_gcp_and_init_client():
    global client
    
    # 1. 處理權限認證 (Render 專用技巧)
    # 將環境變數中的 JSON 字串轉存為暫存檔案
    json_creds = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if json_creds:
        try:
            # 建立暫存檔
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp:
                temp.write(json_creds)
                temp_path = temp.name
            
            # 設定環境變數供 SDK 使用
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            print(f"✅ GCP 憑證已載入: {temp_path}")
        except Exception as e:
            print(f"❌ 憑證處理失敗: {e}")
    else:
        print("⚠️ 警告: 未偵測到 GOOGLE_CREDENTIALS_JSON，將嘗試使用預設環境權限 (本機開發可忽略)")

    # 2. 初始化 Client
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        print(f"✅ Google GenAI Client 初始化成功 (Project: {PROJECT_ID}, Model: {MODEL_ID})")
    except Exception as e:
        print(f"❌ Client 初始化失敗: {e}")

# 啟動時執行
setup_gcp_and_init_client()

@app.get("/")
def root():
    return {"status": "Running", "model": MODEL_ID}

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    prompt: str = Form(...)
):
    """
    接收 Client 傳來的圖片與 Prompt，呼叫 Vertex AI
    """
    if not client:
        raise HTTPException(status_code=500, detail="Vertex AI Client 未初始化")

    try:
        # 1. 讀取圖片 Bytes
        image_data = await file.read()
        
        # 2. 判斷 Mime Type (依照你的邏輯)
        mime_type = "image/jpeg"
        if file.filename.lower().endswith(".png"):
            mime_type = "image/png"

        # 3. 準備 Image Part (你的寫法)
        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type=mime_type
        )

        # 4. 設定生成參數 (你的寫法)
        # 注意: thinking_config 是實驗性功能，若報錯請先註解掉
        config = types.GenerateContentConfig(
            temperature=1,
            thinking_config=types.ThinkingConfig(thinking_level="low") 
        )

        # 5. 呼叫 API
        print(f"正在分析圖片: {file.filename} ...")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[image_part, prompt],
            config=config
        )

        # 6. 回傳結果 (需轉換為 JSON 格式)
        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "candidates_tokens": response.usage_metadata.candidates_token_count
            }

        return {
            "filename": file.filename,
            "text": response.text,
            "usage": usage
        }

    except Exception as e:
        print(f"❌ 處理錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))