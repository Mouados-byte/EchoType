# main.py
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from tempfile import NamedTemporaryFile
import logging
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcription API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model
try:
    model = WhisperModel("base", device="cpu", compute_type="int8")
except Exception as e:
    raise

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile):
    try:
        # Read the uploaded file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name

            # Write content to temporary file
            temp_file.write(content)
            temp_file.flush()
            
            segments, info = model.transcribe(temp_path)
            
            # Join all segments text
            full_text = " ".join([segment.text for segment in segments]).strip()
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "text": full_text,
                "language": info.language,
                "segments": [{"text": segment.text, "start": segment.start, "end": segment.end} for segment in segments]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Transcription API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)