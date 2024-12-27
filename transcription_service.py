import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from faster_whisper import WhisperModel
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float

@dataclass
class TranscriptionResult:
    full_text: str
    language: str
    segments: List[TranscriptionSegment]

class WhisperTranscriptionService:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict] = None
    ):

        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {"min_silence_duration_ms": 500}
        
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        try:
            logger.info(f"Starting transcription for file: {file_path}")
            
            segments, info = self.model.transcribe(
                file_path,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters
            )
            
            transcription_segments = []
            full_text_parts = []
            
            for segment in segments:
                transcription_segments.append(
                    TranscriptionSegment(
                        text=segment.text,
                        start=round(segment.start, 2),
                        end=round(segment.end, 2)
                    )
                )
                full_text_parts.append(segment.text)
            
            result = TranscriptionResult(
                full_text=" ".join(full_text_parts).strip(),
                language=info.language,
                segments=transcription_segments
            )
            
            logger.info("Transcription completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def transcribe_bytes(self, audio_bytes: bytes, temp_file_suffix: str = '.wav') -> TranscriptionResult:
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=temp_file_suffix) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                result = self.transcribe_file(temp_path)
                
                os.unlink(temp_path)
                return result
                
        except Exception as e:
            logger.error(f"Failed to transcribe audio bytes: {str(e)}")
            raise