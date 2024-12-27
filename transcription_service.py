import logging
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from faster_whisper import WhisperModel
import os
import numpy as np
from scipy import signal
import soundfile as sf

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

   def clean_audio(self, audio_data, sample_rate=16000):
       audio = np.array(audio_data)
       
       max_val = np.max(np.abs(audio))
       if max_val > 0:
           audio = audio / max_val
       
       nyquist = sample_rate / 2
       low = 300 / nyquist
       high = 3400 / nyquist
       b, a = signal.butter(4, [low, high], btype='band')
       filtered = signal.filtfilt(b, a, audio)
       
       if len(filtered) > 0:
           filtered = signal.medfilt(filtered, kernel_size=3)
       
       return filtered

   def transcribe_file(self, file_path: str, language: str = None) -> TranscriptionResult:
       try:
           audio, sr = sf.read(file_path)
           cleaned_audio = self.clean_audio(audio, sr)
           
           with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
               sf.write(temp_file.name, cleaned_audio, sr)
               
               segments, info = self.model.transcribe(
                   temp_file.name,
                   beam_size=self.beam_size,
                   vad_filter=self.vad_filter,
                   vad_parameters=self.vad_parameters,
                   language=language
               )
               
               os.unlink(temp_file.name)
               
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
               
               return TranscriptionResult(
                   full_text=" ".join(full_text_parts).strip(),
                   language=info.language,
                   segments=transcription_segments
               )
               
       except Exception as e:
           logger.error(f"Transcription failed: {str(e)}")
           raise

   def transcribe_bytes(self, audio_bytes: bytes, language: str = None, temp_file_suffix: str = '.wav') -> TranscriptionResult:
       try:
           with tempfile.NamedTemporaryFile(delete=False, suffix=temp_file_suffix) as temp_file:
               temp_path = temp_file.name
               temp_file.write(audio_bytes)
               temp_file.flush()
               
               result = self.transcribe_file(temp_path, language)
               
               os.unlink(temp_path)
               return result
               
       except Exception as e:
           logger.error(f"Failed to transcribe audio bytes: {str(e)}")
           raise