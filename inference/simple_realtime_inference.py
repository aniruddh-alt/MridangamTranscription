#!/usr/bin/env python3
"""
Simple Real-Time Mridangam Inference Pipeline

A streamlined real-time audio inference system for mridangam stroke classification.
This script provides essential functionality for:
- Real-time audio capture from microphone
- Onset detection for mridangam strokes  
- CNN model inference using modelOutput.pth
- Simple console output of detected strokes

Features:
- Uses yield for streaming results
- CUDA GPU acceleration
- Minimal dependencies
- Easy to understand and modify
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import pyaudio
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Generator, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MridangamCNN(nn.Module):
    def __init__(self, n_mels = 128, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(

            # input shape: (1, 128, 128)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (32, 128, 128)

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(p=0.3),
            # output shape: (32, 64, 64)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (64, 64, 64)

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(p=0.3),
            # output shape: (64, 32, 32)

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (64, 32, 32)

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(p=0.4),
            # output shape: (64, 16, 16)
            nn.AdaptiveAvgPool2d((1, None)),  # Then reduce spatial

        )

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # output shape: (64, 1, 16)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # output shape: (64, 1, 1)
            nn.Dropout(p=0.4),
            nn.Flatten(),  # output shape: (64)
            nn.Linear(64, num_classes),  # output shape: (num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(2)
        x = self.classifier(x)
        return x



class SimpleAudioListener:
    """Simple real-time audio capture and processing"""
    
    def __init__(self, sample_rate=22050, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.is_recording = True
        
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        def record_audio():
            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_queue.put(audio_chunk)
                except Exception as e:
                    print(f"Audio recording error: {e}")
                    break
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"🎤 Started recording at {self.sample_rate}Hz...")
        
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        self.p.terminate()
        print("🛑 Stopped recording")
        
    def get_audio_chunk(self, timeout=0.1):
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def detect_onset(audio_buffer: np.ndarray, sr: float) -> Optional[float]:
    """
    Simple onset detection for mridangam strokes
    Returns onset time if detected, None otherwise
    """
    if len(audio_buffer) < sr * 0.1:  # Need at least 100ms of audio
        return None
        
    try:
        # Use spectral flux for onset detection
        onsets = librosa.onset.onset_detect(
            y=audio_buffer, 
            sr=sr, 
            units='time',
            pre_max=3, 
            post_max=3, 
            pre_avg=3, 
            post_avg=3,
            delta=0.2,
            wait=5
        )
        
        if len(onsets) > 0:
            return onsets[-1]  # Return most recent onset
            
    except Exception as e:
        print(f"Onset detection error: {e}")
        
    return None


def extract_mel_spectrogram(audio: np.ndarray, sr: float, n_mels=128, target_length=128) -> np.ndarray:
    """
    Extract mel spectrogram from audio segment
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            hop_length=512,
            win_length=1024,
            n_fft=2048
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to target length
        if mel_spec_db.shape[1] < target_length:
            pad_width = target_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_length]
            
        return mel_spec_db
        
    except Exception as e:
        print(f"Mel spectrogram extraction error: {e}")
        return None


def load_model(model_path: str, device: torch.device) -> MridangamCNN:
    """
    Load the trained model from modelOutput.pth
    """
    try:
        # Create model instance
        model = MridangamCNN(n_mels=128, num_classes=10)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"🔧 Using device: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


class SimpleStrokeDetector:
    """Simple real-time stroke detection system"""
    
    def __init__(self, model_path: str, class_names=None):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = load_model(model_path, self.device)
        
        # Default class names (update these based on your training data)
        self.class_names = class_names or [
            'Na', 'Ka', 'Dhi', 'Mi', 'Ki', 'Ta', 'Tha', 'Dhim', 'Tom', 'Arai'
        ]
        
        # Audio settings
        self.sample_rate = 22050
        self.buffer_duration = 2.0  # seconds
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size)
        
        # Initialize audio listener
        self.audio_listener = SimpleAudioListener(sample_rate=self.sample_rate)
        
        print(f"🎵 Stroke detector initialized")
        print(f"📊 Classes: {', '.join(self.class_names)}")
        
    def predict_stroke(self, audio_segment: np.ndarray) -> Tuple[str, float]:
        """
        Predict stroke from audio segment
        Returns (class_name, confidence)
        """
        try:
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(audio_segment, self.sample_rate)
            if mel_spec is None:
                return "Unknown", 0.0
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
            input_tensor = input_tensor.to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            predicted_stroke = self.class_names[predicted_class]
            return predicted_stroke, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0
    
    def extract_stroke_segment(self, onset_time: float) -> np.ndarray:
        """Extract audio segment around detected onset"""
        # Get segment around onset (50ms before, 150ms after)
        pre_onset = int(0.05 * self.sample_rate)
        post_onset = int(0.15 * self.sample_rate)
        
        onset_sample = int(onset_time * self.sample_rate)
        start_idx = max(0, len(self.audio_buffer) - onset_sample - pre_onset)
        end_idx = min(len(self.audio_buffer), len(self.audio_buffer) - onset_sample + post_onset)
        
        return self.audio_buffer[start_idx:end_idx]
    
    def run_detection(self) -> Generator[Tuple[str, float, float], None, None]:
        """
        Run real-time stroke detection
        Yields (stroke_name, confidence, timestamp) for each detected stroke
        """
        self.audio_listener.start_recording()
        
        try:
            last_detection_time = 0
            detection_cooldown = 0.2  # Minimum time between detections (seconds)
            
            print("🎤 Listening for mridangam strokes... (Press Ctrl+C to stop)")
            print("=" * 50)
            
            while True:
                # Get new audio chunk
                chunk = self.audio_listener.get_audio_chunk(timeout=0.05)
                if chunk is not None:
                    # Update circular buffer
                    self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
                    self.audio_buffer[-len(chunk):] = chunk
                
                # Check for onset in recent audio
                current_time = time.time()
                if current_time - last_detection_time > detection_cooldown:
                    
                    # Look for onset in recent audio
                    recent_audio = self.audio_buffer[-int(0.5 * self.sample_rate):]  # Last 500ms
                    onset_time = detect_onset(recent_audio, self.sample_rate)
                    
                    if onset_time is not None:
                        # Extract segment around onset
                        stroke_segment = self.extract_stroke_segment(onset_time)
                        
                        if len(stroke_segment) > 0:
                            # Predict stroke
                            stroke_name, confidence = self.predict_stroke(stroke_segment)
                            
                            # Only report high-confidence predictions
                            if confidence > 0.5:
                                timestamp = current_time
                                last_detection_time = current_time
                                yield stroke_name, confidence, timestamp
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        except Exception as e:
            print(f"❌ Detection error: {e}")
        finally:
            self.audio_listener.stop_recording()


def main():
    """Main function to run the simple stroke detector"""
    
    # Path to your trained model
    model_path = r"c:\Users\aniru\Desktop\MridangamTranscription\inference\model_weights\modelOutput.pth"
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure modelOutput.pth is in the correct location.")
        return
    
    try:
        # Initialize detector
        detector = SimpleStrokeDetector(model_path)
        
        # Run detection loop
        for stroke_name, confidence, timestamp in detector.run_detection():
            # Format timestamp
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            # Print detection with color coding
            confidence_bar = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            print(f"{time_str} | {confidence_bar} {stroke_name:>8} | {confidence:.2%}")
            
    except Exception as e:
        print(f"❌ Error running detector: {e}")


if __name__ == "__main__":
    main()
