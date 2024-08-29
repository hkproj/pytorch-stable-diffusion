# sd_api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sd import generate_image
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    try:
        output_image = generate_image(prompt)
        img = Image.fromarray(output_image)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
