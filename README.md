# Speech Translation Implementation Guide

## Client Side (index.html)

The main HTML file contains three core components:

1. Audio Capture & Processing
- Handles microphone input
- Processes audio chunks
- Converts to WAV format
- Sample rate: 16000Hz
- Buffer size: 2048

2. WebSocket Communication 
- Manages real-time streaming
- Handles chunked file uploads
- Processes server responses
- Automatic reconnection

3. File Upload System
- Chunked file processing
- Progress tracking
- Base64 conversion

## Server Side (FastAPI)
![diagram](https://github.com/user-attachments/assets/ec5712b8-e8b3-4ec0-a437-f5199bec9915)


### WebSocket Endpoints

1. Real-time Speech (/ws)
Input:
{
    "type": "audio_chunk",
    "data": "data:audio/wav;base64,...",
    "language": "en"
}

Output:
{
    "type": "transcription",
    "text": "Transcribed text..."
}

2. File Upload (/ws/transcribe)
Input:
{
    "type": "file_chunk",
    "data": "data:audio/...;base64,...",
    "chunk_number": 1,
    "total_chunks": 10,
    "language": "en"
}

Responses:
- Chunk Receipt:
{
    "type": "chunk_received",
    "chunk": 1
}

- Segment Update:
{
    "type": "segment",
    "number": 1,
    "text": "Segment text...",
    "start": 0.0,
    "end": 2.5
}

- Complete:
{
    "type": "complete",
    "text": "Full transcription..."
}

### HTTP Endpoint

POST /transcribe
- Input: multipart/form-data (audio file)
- Output: {"text": "Transcribed text..."}
- Error: {"success": false, "error": "Description"}

## Quick Setup

1. Server:
```python
docker build -t whisper-api .
docker run -p 8000:8000 whisper-api
