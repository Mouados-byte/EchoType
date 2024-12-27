from transcription_service import WhisperTranscriptionService
import argparse
import sys

def transcribe_audio_file(file_path: str, model_size: str = "base") -> None:
    try:
        service = WhisperTranscriptionService(
            model_size=model_size,
            device="cpu",
            compute_type="int8"
        )
        
        print(f"\nTranscribing file: {file_path}")
        print("This might take a few moments...\n")
        
        result = service.transcribe_file(file_path)
        
        print("=" * 50)
        print("TRANSCRIPTION RESULTS")
        print("=" * 50)
        print(f"\nDetected Language: {result.language}")
        print("\nFull Text:")
        print("-" * 50)
        print(result.full_text)
        print("\nDetailed Segments:")
        print("-" * 50)
        for segment in result.segments:
            print(f"\n[{segment.start}s -> {segment.end}s]")
            print(f"{segment.text}")
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper")
    parser.add_argument("file_path", help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="base", 
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model size to use (default: base)")
    
    args = parser.parse_args()
    
    transcribe_audio_file(args.file_path, args.model)