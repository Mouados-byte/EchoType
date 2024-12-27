import tempfile
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import json
import base64
from transcription_service import WhisperTranscriptionService
from websocket_manager import ConnectionManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Live Speech Translation")
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
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                if message["type"] == "audio":
                    # Handle live audio chunks
                    result = await manager.process_audio_chunk(
                        websocket,
                        message["data"],
                        whisper_service
                    )
                    if result:
                        await websocket.send_json({
                            "type": "transcription",
                            "original": result + " "
                        })

                elif message["type"] == "upload_chunk":
                    # Handle file upload chunks
                    try:
                        chunk_data = base64.b64decode(message["data"].split(",")[1])
                        await manager.add_chunk(websocket, chunk_data)
                        
                        chunk_number = message["chunk_number"]
                        total_chunks = message["total_chunks"]
                        
                        # If this is the last chunk, process the complete file
                        if chunk_number == total_chunks:
                            file_data = await manager.get_complete_file(websocket)
                            if file_data:
                                # Create temporary file and process it
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                                    temp_file.write(file_data)
                                    temp_file.flush()
                                    
                                    try:
                                        result = whisper_service.transcribe_file(temp_file.name)
                                        if result and result.full_text.strip():
                                            await websocket.send_json({
                                                "type": "transcription",
                                                "original": result.full_text + " "
                                            })
                                    finally:
                                        Path(temp_file.name).unlink(missing_ok=True)
                    
                    except Exception as e:
                        logger.error(f"File upload error: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON message")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        manager.cleanup()
        import gc
        gc.collect()

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