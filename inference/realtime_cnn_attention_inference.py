#!/usr/bin/env python3
"""
Real-Time Mridangam CNN with Attention Inference Pipeline

A comprehensive real-time audio inference system for mridangam stroke classification
using the CNN with Attention model. This script provides:
- Real-time audio capture from microphone
- Onset detection optimized for mridangam strokes  
- CNN with Attention model inference
- Visual feedback and confidence scores
- Same preprocessing pipeline as training

Features:
- Uses the exact CNN with Attention architecture from training
- Per-mel-bin normalization with training statistics
- CUDA GPU acceleration
- Real-time visual feedback
- Configurable confidence thresholds
- Support for different model checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pyaudio
import threading
import queue
import time
import pickle
import os
from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, List
import warnings
from collections import deque
import sys
import json

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import preprocessing functions
try:
    from dataset.data_processing.data_preparation import get_audio, get_onset, get_window, get_mel_spectrogram
except ImportError:
    print("Warning: Could not import preprocessing functions. Using local implementations.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# CNN with Attention Model Architecture
class FrequencyAttentivePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_mean = x.mean(dim=-1) # (batch, channels, freq)
        attentive_pooling = self.attention(x_mean)  # (batch, 1, freq)
        att_weights = torch.softmax(attentive_pooling, dim=-1)  # Normalize weights
        x_weighted = x_mean * att_weights  # (batch, channels, freq)
        return x_weighted.sum(dim=-1)  # (batch, channels)

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W)
        proj_key = self.key(x).view(B, -1, H * W)
        proj_value = self.value(x).view(B, -1, H * W)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = F.softmax(energy / (proj_query.size(1) ** 0.5), dim=-1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attn = SelfAttention2D(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attn(out)
        out += residual
        return self.relu(out)

class MridangamCNN(nn.Module):
    def __init__(self, n_mels=128, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.6),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.6),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.8),
            
            ResidualBlock(channels=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout2d(p=dropout_rate * 0.8),
        )
        
        self.attention_pooling = FrequencyAttentivePooling(in_channels=128)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),              # 0
            nn.Linear(128, 64),                      # 1
            nn.BatchNorm1d(64),                      # 2
            nn.ReLU(inplace=True),                   # 3
            nn.Dropout(p=dropout_rate * 0.6),       # 4
            nn.Linear(64, num_classes)               # 5
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention_pooling(x)
        x = self.classifier(x)
        return x

class RealTimeAudioProcessor:
    """Enhanced real-time audio capture and processing"""
    
    def __init__(self, sample_rate=22050, chunk_size=1024, channels=1, buffer_duration=2.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Circular buffer for continuous audio
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.is_recording = True
        
        try:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=None  # Use default device
            )
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
            print("💡 Try checking your microphone permissions")
            return False
        
        def record_audio():
            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Add to circular buffer
                    self.audio_buffer.extend(audio_chunk)
                    
                    # Also add to queue for processing
                    self.audio_queue.put(audio_chunk)
                    
                except Exception as e:
                    print(f"Audio recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        print(f"🎤 Started recording at {self.sample_rate}Hz...")
        return True
        
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1.0)
        self.p.terminate()
        print("🛑 Stopped recording")
        
    def get_audio_buffer(self) -> np.ndarray:
        """Get current audio buffer as numpy array"""
        if len(self.audio_buffer) > 0:
            return np.array(self.audio_buffer)
        return np.array([])
    
    def get_recent_audio(self, duration: float = 0.5) -> np.ndarray:
        """Get most recent audio of specified duration"""
        samples_needed = int(duration * self.sample_rate)
        buffer_array = self.get_audio_buffer()
        
        if len(buffer_array) >= samples_needed:
            return buffer_array[-samples_needed:]
        else:
            return buffer_array

def get_onset_local(audio: np.ndarray, sr: float) -> Optional[float]:
    """Local onset detection implementation"""
    if len(audio) < sr * 0.1:  # Need at least 100ms
        return None
        
    try:
        # Spectral flux onset detection
        spectral_onsets = librosa.onset.onset_detect(
            y=audio, sr=sr, units='time',
            onset_envelope=librosa.onset.onset_strength(y=audio, sr=sr),
            pre_max=3, post_max=3, pre_avg=3, post_avg=3,
            delta=0.3, wait=5
        )
        
        # Energy-based onset detection
        rms_features = librosa.feature.rms(y=audio)[0]
        rms_diff = np.diff(rms_features, prepend=rms_features[0])
        rms_diff = np.maximum(0, rms_diff)
        
        energy_onsets = librosa.onset.onset_detect(
            onset_envelope=rms_diff,
            sr=sr, units='time',
            pre_max=3, post_max=3, pre_avg=3, post_avg=3,
            delta=0.4, wait=5
        )
        
        # Combine all detected onsets
        all_onsets = np.concatenate([spectral_onsets, energy_onsets])
        
        # Remove duplicates within 50ms window
        if len(all_onsets) > 0:
            all_onsets = np.sort(all_onsets)
            unique_onsets = [all_onsets[0]]
            for onset in all_onsets[1:]:
                if onset - unique_onsets[-1] > 0.05:  # 50ms threshold
                    unique_onsets.append(onset)
            onset_times = np.array(unique_onsets)
        else:
            onset_times = all_onsets
        
        return onset_times[-1] if len(onset_times) > 0 else None
        
    except Exception as e:
        print(f"Onset detection error: {e}")
        return None

def get_window_local(onset: float, audio: np.ndarray, sr: float, 
                     pre_onset: float = 0.05, post_onset: float = 0.15) -> np.ndarray:
    """Local window extraction implementation"""
    duration = pre_onset + post_onset
    window_samples = int(duration * sr)
    
    if onset is None:
        if len(audio) >= window_samples:
            return audio[:window_samples]
        else:
            return np.pad(audio, (0, window_samples - len(audio)), mode='constant')
    
    # Put onset at 25% of window
    pre_onset_duration = duration * 0.25
    
    start_time = max(0, onset - pre_onset_duration)
    end_time = start_time + duration
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Handle edge cases
    if end_sample > len(audio):
        end_sample = len(audio)
        start_sample = max(0, end_sample - window_samples)
    
    # Extract window
    window = audio[start_sample:end_sample]
    
    # Ensure exact duration
    if len(window) < window_samples:
        pad_amount = window_samples - len(window)
        window = np.pad(window, (0, pad_amount), mode='constant')
    elif len(window) > window_samples:
        window = window[:window_samples]
    
    return window

def get_mel_spectrogram_local(audio: np.ndarray, sr: float) -> np.ndarray:
    """Local mel spectrogram extraction implementation"""
    # Dynamic n_fft sizing based on audio length
    max_n_fft = min(512, len(audio))
    n_fft = max(256, max_n_fft)
    hop_length = n_fft // 2
    
    # Ensure audio is long enough for n_fft
    if len(audio) < n_fft:
        pad_amt = n_fft - len(audio)
        audio = np.pad(audio, (0, pad_amt), mode='constant')
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128
    )

    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

class RealTimeStrokeDetector:
    """Real-time mridangam stroke detection with CNN Attention model"""
    
    def __init__(self, model_path: str, mel_stats_path: Optional[str] = None, 
                 class_names: Optional[List[str]] = None, confidence_threshold: float = 0.5,
                 classes_file: Optional[str] = None):
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load mel statistics for normalization
        self.mel_stats = self._load_mel_stats(mel_stats_path)
        
        # Resolve class names
        self.class_names = self._resolve_class_names(model_path=model_path,
                                                    user_class_names=class_names,
                                                    classes_file=classes_file)
            
        self.confidence_threshold = confidence_threshold
        self.last_detection_time = 0
        self.min_detection_interval = 0.1  # Minimum 100ms between detections
        
        # Audio processor
        self.audio_processor = RealTimeAudioProcessor()
        
        print(f"✅ Model loaded successfully")
        print(f"📋 Classes: {self.class_names}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        
    def _load_model(self, model_path: str) -> MridangamCNN:
        """Load the trained CNN with attention model"""
        try:
            # Try to import and use original training model architecture first
            try:
                import sys
                sys.path.append(str(Path(__file__).parent.parent))
                from model_architecture.CNN_with_attention import load_model_for_inference
                print("✅ Using original training model architecture")
                return load_model_for_inference(model_path, self.device)
            except ImportError:
                print("⚠️ Could not import original model, using local implementation")
            
            # Fallback to local implementation
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Infer number of classes from the final layer
            num_classes = 10  # Default
            for key, tensor in state_dict.items():
                if key.endswith('classifier.5.weight'):
                    num_classes = tensor.shape[0]
                    break
            
            # Create model
            model = MridangamCNN(n_mels=128, num_classes=num_classes, dropout_rate=0.5)
            
            # Load state dict with flexible loading
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")
            
            model = model.to(self.device)
            model.eval()
            
            print(f"📊 Model architecture: CNN with Attention ({num_classes} classes)")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _load_classes_from_checkpoint(self, model_path: str) -> Optional[List[str]]:
        """Try to load class names from a checkpoint that includes metadata."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                # Direct key
                if 'classes' in checkpoint and isinstance(checkpoint['classes'], (list, tuple)):
                    return list(checkpoint['classes'])
                # Embedded label encoder
                if 'label_encoder' in checkpoint:
                    le = checkpoint['label_encoder']
                    # Some checkpoints save sklearn LabelEncoder; try to access classes_
                    classes = getattr(le, 'classes_', None)
                    if classes is not None:
                        return list(classes)
        except Exception:
            pass
        return None

    def _load_classes_from_file(self, classes_file: Optional[str]) -> Optional[List[str]]:
        """Load class names from a JSON array or newline-separated text file."""
        if not classes_file:
            return None
        try:
            with open(classes_file, 'r') as f:
                content = f.read().strip()
            # Try JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return data
            except json.JSONDecodeError:
                # Fallback: newline-separated
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                if lines:
                    return lines
        except Exception as e:
            print(f"⚠️ Could not load classes from file '{classes_file}': {e}")
        return None

    def _resolve_class_names(self, model_path: str,
                              user_class_names: Optional[List[str]],
                              classes_file: Optional[str]) -> List[str]:
        """Resolve class names in priority: user > file > checkpoint > default/generic.

        Ensures length matches model output classes.
        """
        # Determine num_classes from model
        num_classes = 10
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor) and key.endswith('classifier.5.weight'):
                    num_classes = tensor.shape[0]
                    break
        except Exception:
            pass

        # 1) Explicit user-provided list
        if user_class_names:
            names = list(user_class_names)
        else:
            # 2) From classes file
            names = self._load_classes_from_file(classes_file)
            if names is None:
                # 3) From checkpoint metadata
                names = self._load_classes_from_checkpoint(model_path)

        # 4) Fallbacks
        if not names:
            # Historically used names (may not match training). Prefer generic if mismatch.
            default_names = ['Ta', 'Tha', 'Dha', 'Na', 'Thi', 'Ki', 'Dhi', 'Tam', 'Dhom', 'Kita']
            names = default_names[:num_classes]
            if len(names) != num_classes:
                names = [f'class_{i}' for i in range(num_classes)]

        # If length mismatch, fix deterministically
        if len(names) != num_classes:
            print(f"⚠️ Classes length ({len(names)}) doesn't match model outputs ({num_classes}). Using generic names.")
            names = [f'class_{i}' for i in range(num_classes)]

        return names
    
    def _load_mel_stats(self, mel_stats_path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
        """Load mel spectrogram normalization statistics"""
        if mel_stats_path and os.path.exists(mel_stats_path):
            try:
                with open(mel_stats_path, 'rb') as f:
                    mel_stats = pickle.load(f)
                print(f"📈 Loaded mel statistics from {mel_stats_path}")
                return mel_stats
            except Exception as e:
                print(f"⚠️ Warning: Could not load mel stats: {e}")
        
        print("⚠️ Warning: No mel statistics loaded. Using default normalization.")
        return None
    
    def _preprocess_audio(self, audio: np.ndarray, sr: float, pre_windowed: bool = False) -> torch.Tensor:
        """Preprocess audio using the same pipeline as training.

        If pre_windowed is True, assumes `audio` is already a 0.2s window around an onset,
        and skips onset/window extraction.
        """
        try:
            # Try to use imported functions, fallback to local implementations
            try:
                from dataset.data_processing.data_preparation import get_onset, get_window, get_mel_spectrogram
                if pre_windowed:
                    mel_spec = get_mel_spectrogram(audio, sr)
                else:
                    onset = get_onset(audio, sr)
                    audio_window = get_window(onset, audio, sr)
                    mel_spec = get_mel_spectrogram(audio_window, sr)
            except (ImportError, NameError):
                # Fall back to local implementations
                if pre_windowed:
                    mel_spec = get_mel_spectrogram_local(audio, sr)
                else:
                    onset = get_onset_local(audio, sr)
                    audio_window = get_window_local(onset, audio, sr)
                    mel_spec = get_mel_spectrogram_local(audio_window, sr)
            
            # Normalize and format
            mel_tensor = self._normalize_and_format(mel_spec)
            
            return mel_tensor
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, 1, 128, 128).to(self.device)

    @staticmethod
    def _validate_segment(audio_segment: np.ndarray, sr: float) -> bool:
        """Basic energy/duration validation to reduce false positives."""
        if audio_segment is None or len(audio_segment) < int(0.05 * sr):
            return False
        rms = float(np.sqrt(np.mean(np.square(audio_segment)))) if len(audio_segment) > 0 else 0.0
        peak = float(np.max(np.abs(audio_segment))) if len(audio_segment) > 0 else 0.0
        if rms < 0.01 or peak < 0.05:
            return False
        return True
    
    def _normalize_and_format(self, mel_spec: np.ndarray, target_length: int = 128) -> torch.Tensor:
        """Normalize per mel-bin and format for CNN architecture"""
        # Ensure consistent time dimension
        n_mels, time_frames = mel_spec.shape
        
        # Pad or truncate to target length
        if time_frames < target_length:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, target_length - time_frames)), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_length]
        
        # Per-frequency normalization
        if self.mel_stats is not None:
            # Normalize each mel bin independently
            mel_spec = (mel_spec - self.mel_stats['mean'][:, np.newaxis]) / (self.mel_stats['std'][:, np.newaxis] + 1e-8)
        
        # Convert to tensor and add batch and channel dimensions
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_frames)
        
        return mel_tensor.to(self.device)
    
    def predict_stroke(self, audio_segment: np.ndarray, sr: float = 22050, pre_windowed: bool = False) -> Tuple[str, float, np.ndarray]:
        """Predict mridangam stroke from audio segment"""
        try:
            # Validate segment
            if not self._validate_segment(audio_segment, sr):
                return "Weak_Signal", 0.0, np.zeros(len(self.class_names))
            # Preprocess audio
            mel_tensor = self._preprocess_audio(audio_segment, sr, pre_windowed=pre_windowed)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(mel_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_val = confidence.item()
                all_probs = probabilities.cpu().numpy().flatten()
                
                return predicted_class, confidence_val, all_probs
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Unknown", 0.0, np.zeros(len(self.class_names))
    
    def run_detection(self, duration: Optional[float] = None) -> Generator[Tuple[str, float, float, np.ndarray], None, None]:
        """Run real-time stroke detection"""
        if not self.audio_processor.start_recording():
            return
        
        start_time = time.time()
        last_onset_time = 0
        
        try:
            print("🎵 Listening for mridangam strokes... (Press Ctrl+C to stop)")
            print(f"🎯 Confidence threshold: {self.confidence_threshold}")
            print("=" * 60)
            
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get recent audio for onset detection (1.0s buffer)
                recent_audio = self.audio_processor.get_recent_audio(duration=1.0)
                
                if len(recent_audio) > 0:
                    current_time = time.time()
                    
                    # Detect onset in recent audio
                    onset = get_onset_local(recent_audio, self.audio_processor.sample_rate)
                    
                    if onset is not None:
                        # Calculate absolute onset time
                        absolute_onset_time = current_time - 1.0 + onset
                        
                        # Check if this is a new onset (avoid duplicate detections)
                        if absolute_onset_time - last_onset_time > self.min_detection_interval:
                            last_onset_time = absolute_onset_time
                            
                            # Extract exact window around the detected onset from the recent buffer
                            segment_audio = get_window_local(onset, recent_audio, self.audio_processor.sample_rate)
                            
                            if len(segment_audio) > 0:
                                # Predict stroke from the windowed segment
                                predicted_class, confidence, all_probs = self.predict_stroke(
                                    segment_audio, self.audio_processor.sample_rate, pre_windowed=True
                                )
                                
                                # Check confidence threshold
                                if confidence >= self.confidence_threshold:
                                    yield predicted_class, confidence, absolute_onset_time, all_probs
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        finally:
            self.audio_processor.stop_recording()

def run_realtime_detection(model_path: str, mel_stats_path: Optional[str] = None, 
                          class_names: Optional[List[str]] = None,
                          confidence_threshold: float = 0.5,
                          duration: Optional[float] = None,
                          classes_file: Optional[str] = None):
    """Run real-time detection with the specified parameters"""
    try:
        # Create detector
        detector = RealTimeStrokeDetector(
            model_path=model_path,
            mel_stats_path=mel_stats_path,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            classes_file=classes_file
        )
        
        # Detection statistics
        detection_count = 0
        class_counts = {name: 0 for name in detector.class_names}
        
        print(f"\n🚀 Starting real-time mridangam stroke detection...")
        print(f"📁 Model: {os.path.basename(model_path)}")
        if duration:
            print(f"⏱️ Duration: {duration}s")
        print("\n" + "=" * 60)
        
        # Run detection
        for predicted_class, confidence, timestamp, all_probs in detector.run_detection(duration=duration):
            detection_count += 1
            class_counts[predicted_class] += 1
            
            # Format timestamp
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            # Print detection
            print(f"🥁 [{time_str}] {predicted_class:>6} (confidence: {confidence:.3f})")
            
            # Show top 3 predictions if confidence is high
            if confidence > 0.7:
                top_indices = np.argsort(all_probs)[::-1][:3]
                top_predictions = [(detector.class_names[i], all_probs[i]) for i in top_indices]
                print(f"    Top 3: {' | '.join([f'{name}: {prob:.2f}' for name, prob in top_predictions])}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total detections: {detection_count}")
        if detection_count > 0:
            print("\nClass distribution:")
            for class_name, count in class_counts.items():
                if count > 0:
                    percentage = (count / detection_count) * 100
                    print(f"  {class_name:>6}: {count:3d} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()

def run_file_inference(wav_path: str, model_path: str, mel_stats_path: Optional[str] = None,
                       class_names: Optional[List[str]] = None, classes_file: Optional[str] = None,
                       confidence_threshold: float = 0.5) -> Dict[str, any]:
    """Offline evaluation on a WAV file with training-aligned preprocessing."""
    # Create detector without starting mic
    detector = RealTimeStrokeDetector(
        model_path=model_path,
        mel_stats_path=mel_stats_path,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        classes_file=classes_file
    )
    # Load audio
    try:
        y, sr = librosa.load(wav_path, sr=22050)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio '{wav_path}': {e}")

    # Find onsets using training-like approach
    try:
        # Use multiple onsets across whole file
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, units='time',
            onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
            pre_max=3, post_max=3, pre_avg=3, post_avg=3,
            delta=0.3, wait=5
        )
    except Exception:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    results = []
    for onset in onsets:
        try:
            try:
                from dataset.data_processing.data_preparation import get_window
                segment = get_window(onset, y, sr)
            except Exception:
                segment = get_window_local(onset, y, sr)
            pred_class, conf, probs = detector.predict_stroke(segment, sr, pre_windowed=True)
            results.append((onset, pred_class, conf, probs))
        except Exception as e:
            print(f"Segment error at onset {onset:.3f}: {e}")

    # Summary
    summary = {}
    for _, cls, conf, _ in results:
        if conf >= confidence_threshold:
            summary[cls] = summary.get(cls, 0) + 1

    print("\n" + "=" * 60)
    print(f"Offline file inference: {os.path.basename(wav_path)}")
    print("=" * 60)
    for onset, cls, conf, _ in results:
        print(f"  {onset:6.3f}s -> {cls:>6} (conf: {conf:.3f})")
    print("\nHigh-confidence summary (>= {0:.2f}):".format(confidence_threshold))
    for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:>6}: {v}")

    return {
        'results': results,
        'summary': summary,
        'classes': detector.class_names
    }

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Mridangam Stroke Detection with CNN Attention")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--mel-stats", "-s", help="Path to mel statistics file")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--duration", "-d", type=float, help="Detection duration in seconds")
    parser.add_argument("--classes", help="Comma-separated list of class names")
    parser.add_argument("--classes-file", help="Path to a JSON or newline-separated file with class names")
    parser.add_argument("--wav", help="Run offline inference on a WAV file instead of microphone")
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = None
    if args.classes:
        class_names = [name.strip() for name in args.classes.split(',')]
    
    # Run offline or realtime
    if args.wav:
        run_file_inference(
            wav_path=args.wav,
            model_path=args.model,
            mel_stats_path=args.mel_stats,
            class_names=class_names,
            classes_file=args.classes_file,
            confidence_threshold=args.confidence
        )
    else:
        run_realtime_detection(
            model_path=args.model,
            mel_stats_path=args.mel_stats,
            class_names=class_names,
            confidence_threshold=args.confidence,
            duration=args.duration,
            classes_file=args.classes_file
        )

if __name__ == "__main__":
    main() 