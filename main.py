from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from transcription_service import WhisperTranscriptionService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcription API")

# Mount static directory and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize transcription service
transcription_service = WhisperTranscriptionService(
    model_size="base",
    device="cpu",
    compute_type="int8"
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    try:
        logger.info(f"Received file: {file.filename}")
        content = await file.read()
        
        result = transcription_service.transcribe_bytes(content)
        
        return {
            "text": result.full_text,
            "language": result.language,
            "segments": [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                }
                for segment in result.segments
            ]
        }
            
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)