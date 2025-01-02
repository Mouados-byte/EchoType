import os
import tempfile
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import json
import base64

import numpy as np
import soundfile
from transcription_service import WhisperTranscriptionService
from websocket_manager import ConnectionManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EchoType")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
whisper_service = WhisperTranscriptionService(
    model_size="small",
    device="cpu",
    compute_type="int8"
)

# Initialize connection manager
manager = ConnectionManager()  # Adjust workers based on your CPU

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "audio_chunk":
                try:
                    # Decode the base64 audio chunk
                    chunk_data = base64.b64decode(data["data"].split(",")[1])
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(chunk_data)
                        temp_file.flush()
                        
                        try:
                            # Transcribe individual chunk
                            segments, info = whisper_service.model.transcribe(temp_path, language=data["language"])
                            full_text = " ".join(segment.text for segment in segments).strip() or ""

                            await websocket.send_json({
                                "type": "transcription",
                                "text": full_text
                            })

                        finally:
                            # Clean up temporary file
                            os.unlink(temp_path)
                
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Perform any necessary cleanup
        import gc
        gc.collect()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            # Write content to temporary file
            temp_file.write(content)
            temp_file.flush()
            
            # result = whisper_service.transcribe_file(temp_file.name, "fr")
            segments , info = whisper_service.model.transcribe(temp_path)
            
            # Join all segments text
            full_text = " ".join([segment.text for segment in segments]).strip()
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "text": full_text
            }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    file_chunks = []
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "file_chunk":
                chunk_data = base64.b64decode(data["data"].split(",")[1])
                file_chunks.append(chunk_data)
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "chunk_received",
                    "chunk": data["chunk_number"]
                })
                
                # If this is the last chunk, process the complete file
                if data["chunk_number"] == data["total_chunks"]:
                    # Combine all chunks
                    complete_file = b''.join(file_chunks)
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(complete_file)
                        temp_file.flush()
                        
                        try:
                            # Transcribe and handle generator
                            segments, info = whisper_service.model.transcribe(temp_path)
                            
                            # Process segments from generator
                            segment_list = []
                            for i, segment in enumerate(segments, 1):
                                segment_list.append(segment)
                                await websocket.send_json({
                                    "type": "segment",
                                    "number": i,
                                    "text": segment.text,
                                    "start": segment.start,
                                    "end": segment.end
                                })
                            
                            # Send complete transcription
                            full_text = " ".join(segment.text for segment in segment_list).strip()
                            await websocket.send_json({
                                "type": "complete",
                                "text": full_text
                            })
                            
                        finally:
                            os.unlink(temp_path)
                            file_chunks.clear()
                            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws='auto'
    )