import torch
import torch.nn.functional as F
import librosa
import numpy as np
import pyaudio
import threading
import queue
import time
import argparse
import json
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict

from mridangam_transcription.models.cnn_attention import MridangamCNN
from mridangam_transcription.audio.preprocess import get_onset, get_window, get_mel_spectrogram

class RealTimeTranscriber:
    def __init__(self, model_path: str, device: str = 'auto', 
                 confidence_threshold: float = 0.5, 
                 mapping_path: Optional[str] = None):
        
        self.device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        self.mel_stats = checkpoint['mel_stats']
        
        # Initialize model
        self.model = MridangamCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        
        # Load mapping if provided
        self.mapping = None
        if mapping_path:
            with open(mapping_path, 'r') as f:
                self.mapping = json.load(f)
                
        self.sample_rate = 22050
        self.buffer_duration = 2.0
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        self.is_running = False
        self.min_detection_interval = 0.1 # seconds
        self.last_detection_time = 0

    def predict(self, audio_segment: np.ndarray) -> Optional[Dict]:
        """Predict stroke from audio segment."""
        # Energy gate
        rms = np.sqrt(np.mean(audio_segment**2))
        if rms < 0.01:
            return None
            
        # Preprocess
        mel_spec = get_mel_spectrogram(audio_segment, self.sample_rate)
        
        # Normalization
        mean = np.array(self.mel_stats['mean'])
        std = np.array(self.mel_stats['std'])
        mel_spec = (mel_spec - mean[:, np.newaxis]) / (std[:, np.newaxis] + 1e-8)
        
        # Tensor formatting
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Fixed length padding
        target_len = 128
        if mel_tensor.shape[3] < target_len:
            mel_tensor = F.pad(mel_tensor, (0, target_len - mel_tensor.shape[3]))
        else:
            mel_tensor = mel_tensor[:, :, :, :target_len]
            
        with torch.no_grad():
            outputs = self.model(mel_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)
            
        conf_val = confidence.item()
        if conf_val < self.confidence_threshold:
            return None
            
        stroke = self.classes[idx.item()]
        result = {
            "stroke": stroke,
            "confidence": conf_val,
            "timestamp": time.time()
        }
        
        if self.mapping and stroke in self.mapping:
            result["solkattu"] = self.mapping[stroke]
            
        return result

    def process_wav(self, wav_path: str, output_mode: str = 'both'):
        """Process a WAV file offline."""
        print(f"Processing WAV: {wav_path}")
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)
        
        # Simple onset detection on the whole file
        onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time', delta=0.3)
        
        results = []
        for onset_time in onsets:
            # Extract window
            window = get_window(onset_time, audio, sr)
            res = self.predict(window)
            if res:
                res["timestamp"] = onset_time # Overwrite with file timestamp
                results.append(res)
                self._print_result(res, output_mode)
        return results

    def _print_result(self, res: Dict, mode: str):
        ts = time.strftime("%H:%M:%S", time.localtime(res["timestamp"]))
        if "timestamp" < 3600: # Probably relative file time
            ts = f"{res['timestamp']:6.3f}s"
            
        stroke_str = f"{res['stroke']:>8}"
        conf_str = f"({res['confidence']:.2f})"
        
        if mode == 'strokes':
            print(f"[{ts}] {stroke_str} {conf_str}")
        elif mode == 'solkattu':
            print(f"[{ts}] {res.get('solkattu', '?'):>4} {conf_str}")
        else: # both
            solkattu = res.get('solkattu', '?')
            print(f"[{ts}] {stroke_str} -> {solkattu:>4} {conf_str}")

    def run_mic(self, output_mode: str = 'both'):
        """Run real-time transcription from microphone."""
        p = pyaudio.PyAudio()
        chunk_size = 1024
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        self.is_running = True
        print(f"🎤 Listening... (Threshold: {self.confidence_threshold})")
        print("Press Ctrl+C to stop.")
        
        try:
            while self.is_running:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio_buffer.extend(audio_chunk)
                
                # Periodically check for onsets in the last 0.5s
                now = time.time()
                if now - self.last_detection_time > self.min_detection_interval:
                    buf_arr = np.array(self.audio_buffer)
                    if len(buf_arr) >= int(0.5 * self.sample_rate):
                        recent = buf_arr[-int(0.5 * self.sample_rate):]
                        onset = get_onset(recent, self.sample_rate)
                        
                        if onset is not None:
                            # Extract window around onset
                            window = get_window(onset, recent, self.sample_rate)
                            res = self.predict(window)
                            if res:
                                self._print_result(res, output_mode)
                                self.last_detection_time = now
                                
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.is_running = False
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Mridangam Transcription")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--input", choices=['mic', 'wav'], default='mic', help="Input source")
    parser.add_argument("--wav", type=str, help="Path to WAV file (if input is 'wav')")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", choices=['strokes', 'solkattu', 'both'], default='both', help="Output format")
    parser.add_argument("--mapping", type=str, help="Path to label mapping JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, mps, auto)")
    
    args = parser.parse_args()
    
    transcriber = RealTimeTranscriber(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        mapping_path=args.mapping
    )
    
    if args.input == 'wav':
        if not args.wav:
            print("Error: --wav path required for wav input.")
        else:
            transcriber.process_wav(args.wav, output_mode=args.output)
    else:
        transcriber.run_mic(output_mode=args.output)

