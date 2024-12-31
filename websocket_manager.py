import asyncio
import io
import tempfile
from typing import Dict, List, Optional
import wave
from fastapi import WebSocket, logger
import numpy as np


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.file_chunks: dict = {}
        self.audio_buffers: Dict[int, io.BytesIO] = {}
        self.processing_locks: Dict[int, asyncio.Lock] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        conn_id = id(websocket)
        self.active_connections.append(websocket)
        self.file_chunks[conn_id] = []
        self.audio_buffers[conn_id] = io.BytesIO()
        self.processing_locks[conn_id] = asyncio.Lock()

    def disconnect(self, websocket: WebSocket):
        conn_id = id(websocket)
        if conn_id in self.active_connections:
            self.active_connections.remove(websocket)
        if conn_id in self.file_chunks:
            del self.file_chunks[conn_id]
        if conn_id in self.audio_buffers:
            self.audio_buffers[conn_id].close()
            del self.audio_buffers[conn_id]
        if conn_id in self.processing_locks:
            del self.processing_locks[conn_id]

    async def process_audio_chunk(self, websocket: WebSocket, audio_data: List[float],
                                transcription_service, language: str = None) -> Optional[str]:
        try:
            audio_array = np.array(audio_data, dtype=np.float32)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())
                
                result = transcription_service.transcribe_file(temp_wav.name, language)
                return result.full_text if result and result.full_text.strip() else None

        except Exception as e:
            return None

    def cleanup(self):
        """Clean up resources."""
        self.audio_buffers.clear()
        self.file_chunks.clear()
        self.processing_locks.clear()