from collections import deque
import io
import os
import tempfile
import time
from typing import Optional
import wave
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import json
import base64
import torch
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

whisper_service = WhisperTranscriptionService(
    model_size="small",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8",
)

audio_buffer = []

# Initialize connection manager
manager = ConnectionManager()  # Adjust workers based on your CPU

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload_form", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    audio_buffer = deque(maxlen=10000)
    is_first_chunk = True
    wav_params = None
    sequence = 0
    transcripted_text = ""
    last_reset_time = time.time()
    
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "audio_chunk":
                try:
                    # Decode the base64 audio chunk
                    chunk_data = base64.b64decode(data["data"].split(",")[1])
                    
                    # Handle WAV header for first chunk
                    if is_first_chunk:
                        with io.BytesIO(chunk_data) as wav_io:
                            with wave.open(wav_io, 'rb') as wav_file:
                                wav_params = wav_file.getparams()
                                # Skip WAV header (44 bytes for standard WAV)
                                wav_io.seek(44)
                                raw_audio = wav_io.read()
                        audio_buffer.append(raw_audio)
                        is_first_chunk = False
                    else:
                        # For subsequent chunks, assume raw audio data
                        audio_buffer.append(chunk_data)
                    
                    # Create a new WAV file with proper header
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(chunk_data)
                        temp_file.flush()
                        with wave.open(temp_file.name, 'wb') as wav_out:
                            if wav_params:
                                wav_out.setparams(wav_params)
                                wav_out.writeframes(b''.join(audio_buffer))
                        
                        try:
                            # Transcribe
                            segments = whisper_service.model.transcribe(temp_path, language=data["language"])
                            transcripted_text += segments["text"]
                            
                            old_length_buffer = len(audio_buffer)
                            is_end_of_sentence = should_reset_state(segments, last_reset_time=last_reset_time)
                            if is_end_of_sentence:
                                last_reset_time = time.time()
                            
                            await websocket.send_json({
                                "type": "valid_transcription",
                                "text": segments["text"],
                                "language": segments["language"],
                                "segments": segments["segments"],
                                "sequence": sequence,
                                "sequence_step": len(audio_buffer),
                                "is_final": old_length_buffer != len(audio_buffer) and not audio_buffer
                            })
                            
                            # Only clear buffer if we got meaningful transcription
                            if is_end_of_sentence:
                                audio_buffer.clear()
                                is_first_chunk = True  # Reset for next sequence
                                sequence += 1

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
        audio_buffer.clear()
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
            temp_file.write(content)
            temp_file.flush()

            segments = whisper_service.model.transcribe(temp_path)
            
            # Join all segments text
            full_text = segments["text"]
            
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
                
                await websocket.send_json({
                    "type": "chunk_received",
                    "chunk": data["chunk_number"]
                })
                
                if data["chunk_number"] == data["total_chunks"]:
                    complete_file = b''.join(file_chunks)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(complete_file)
                        temp_file.flush()
                        
                        try:
                            # Transcribe and handle generator
                            result = whisper_service.model.transcribe(temp_path)
                            full_text = result["text"]
                            
                            # Split the text into segments at periods and filter out short segments
                            segments = [s.strip() for s in full_text.split('.') if s.strip() and len(s.strip()) > 2]
                            
                            # Send each segment separately
                            for i, segment_text in enumerate(segments, 1):
                                await websocket.send_json({
                                    "type": "segment",
                                    "number": i,
                                    "text": segment_text + '.'
                                })
                            
                            # Send complete transcription
                            await websocket.send_json({
                                "type": "complete",
                                "text": full_text
                            })
                            
                        finally:
                            os.unlink(temp_path)
                            file_chunks.clear()
                            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        
def should_reset_state(segments, last_reset_time, MIN_SEGMENT_DURATION=1, MAX_SEGMENT_DURATION=5):
    text = segments["text"].strip()
    current_time = time.time()
    last_reset_time = last_reset_time or current_time
    time_since_last_reset = current_time - last_reset_time
    
    # Force reset if it's been too long
    if time_since_last_reset >= MAX_SEGMENT_DURATION:
        return True
        
    # Don't allow reset if it's too soon
    if time_since_last_reset < MIN_SEGMENT_DURATION:
        return False
        
    # Normal sentence boundary detection
    if ("." in text and "..." not in text):  # Your original condition
        return True
        
    return False

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