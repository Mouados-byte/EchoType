from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import tempfile
import wave
from fastapi import WebSocket
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self, max_workers: int = 3):
        self.active_connections: list[WebSocket] = []
        self.file_chunks: dict = {}
        self.audio_buffers: Dict[int, io.BytesIO] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
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
        self.active_connections.remove(websocket)
        if conn_id in self.file_chunks:
            del self.file_chunks[conn_id]
        if conn_id in self.audio_buffers:
            self.audio_buffers[conn_id].close()
            del self.audio_buffers[conn_id]
        if conn_id in self.processing_locks:
            del self.processing_locks[conn_id]

    async def add_chunk(self, websocket: WebSocket, chunk: bytes) -> bool:
        """Add a file chunk to the buffer with async lock."""
        conn_id = id(websocket)
        async with self.processing_locks[conn_id]:
            try:
                self.audio_buffers[conn_id].write(chunk)
                self.file_chunks[conn_id].append(chunk)
                return True
            except Exception as e:
                logger.error(f"Error adding chunk: {str(e)}")
                return False

    async def get_complete_file(self, websocket: WebSocket) -> Optional[bytes]:
        """Get the complete file from chunks with async handling."""
        conn_id = id(websocket)
        async with self.processing_locks[conn_id]:
            try:
                # Use the pre-allocated BytesIO buffer
                buffer = self.audio_buffers[conn_id]
                complete_file = buffer.getvalue()
                
                # Clear the buffer for next use
                buffer.seek(0)
                buffer.truncate()
                self.file_chunks[conn_id] = []
                
                return complete_file
            except Exception as e:
                logger.error(f"Error getting complete file: {str(e)}")
                return None

    async def process_audio_chunk(self, websocket: WebSocket, audio_data: List[float],
                                transcription_service) -> Optional[str]:
        """Process audio chunk in thread pool."""
        try:
            # Convert audio data to numpy array for faster processing
            audio_array = np.array(audio_data, dtype=np.float32)
            
            def transcribe():
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                        # Write WAV file
                        with wave.open(temp_wav.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(16000)
                            wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())
                        
                        # Transcribe
                        result = transcription_service.transcribe_file(temp_wav.name)
                        return result.full_text if result and result.full_text.strip() else None
                except Exception as e:
                    logger.error(f"Transcription error: {str(e)}")
                    return None

            # Process in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                transcribe
            )
            return result

        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            return None

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
        for buffer in self.audio_buffers.values():
            buffer.close()
        self.audio_buffers.clear()
        self.file_chunks.clear()