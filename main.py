from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import json
import base64
import io
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
    model_size="small",  # Better balance for real-time use
    device="cpu",
    compute_type="int8",  # Memory efficient
    beam_size=5,
    vad_parameters={
        "min_silence_duration_ms": 700,
        "speech_pad_ms": 400
    }
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.file_chunks: dict = {}  # Store chunks for each connection

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.file_chunks[id(websocket)] = []

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if id(websocket) in self.file_chunks:
            del self.file_chunks[id(websocket)]

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    def add_chunk(self, websocket: WebSocket, chunk: bytes):
        self.file_chunks[id(websocket)].append(chunk)

    def get_complete_file(self, websocket: WebSocket) -> bytes:
        chunks = self.file_chunks[id(websocket)]
        complete_file = b''.join(chunks)
        self.file_chunks[id(websocket)] = []  # Clear the chunks
        return complete_file

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                if message["type"] == "upload_chunk":
                    # Decode the base64 chunk
                    try:
                        # Extract the actual base64 data after the comma
                        base64_data = message["data"].split(',')[1]
                        chunk_data = base64.b64decode(base64_data)
                        manager.add_chunk(websocket, chunk_data)
                        
                        chunk_number = message["chunk_number"]
                        total_chunks = message["total_chunks"]
                        
                        # Send acknowledgment
                        await manager.send_message(
                            json.dumps({
                                "type": "upload_progress",
                                "chunk": chunk_number,
                                "total": total_chunks,
                                "progress": (chunk_number / total_chunks) * 100
                            }),
                            websocket
                        )
                        
                        # If this is the last chunk, process the complete file
                        if chunk_number == total_chunks:
                            await manager.send_message(
                                json.dumps({
                                    "type": "status",
                                    "message": "Starting transcription..."
                                }),
                                websocket
                            )
                            
                            # Get the complete file and process it
                            complete_file = manager.get_complete_file(websocket)
                            result = transcription_service.transcribe_bytes(complete_file)
                            
                            # Send the transcription result
                            await manager.send_message(
                                json.dumps({
                                    "type": "transcription_complete",
                                    "result": {
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
                                }),
                                websocket
                            )
                    
                    except base64.binascii.Error as e:
                        await manager.send_message(
                            json.dumps({
                                "type": "error",
                                "message": "Invalid file chunk encoding"
                            }),
                            websocket
                        )
                
            except json.JSONDecodeError:
                await manager.send_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid message format"
                    }),
                    websocket
                )
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await manager.send_message(
                    json.dumps({
                        "type": "error",
                        "message": str(e)
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for HTTPS/WSS in production
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws='auto'
    )