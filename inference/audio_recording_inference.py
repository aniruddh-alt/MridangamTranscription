#!/usr/bin/env python3
"""
Audio Recording and Segmentation Inference Pipeline

A comprehensive pipeline for mridangam stroke detection that:
1. Records audio to a file
2. Segments individual strokes using advanced onset detection
3. Processes each segment exactly like the training pipeline
4. Runs inference on each segment

This approach is more accurate than real-time processing because:
- Uses the exact training preprocessing pipeline
- Better stroke segmentation on complete audio
- Eliminates real-time processing artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pyaudio
import wave
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import sys
import warnings
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import training preprocessing functions
try:
    from dataset.data_processing.data_preparation import get_audio, get_onset, get_window, get_mel_spectrogram
    print("✅ Using original training preprocessing functions")
except ImportError:
    print("⚠️ Could not import training preprocessing functions")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# CNN with Attention Model Architecture (same as training)
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
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention_pooling(x)
        x = self.classifier(x)
        return x

class AudioRecorder:
    """Records audio to a file for later processing"""
    
    def __init__(self, sample_rate=22050, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.frames = []
        self.is_recording = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def start_recording(self, filename: str, duration: Optional[float] = None):
        """Start recording audio to a file"""
        self.frames = []
        self.is_recording = True
        
        try:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"🎤 Recording started...")
            if duration:
                print(f"⏱️ Recording for {duration} seconds")
            else:
                print("⏱️ Press Ctrl+C to stop recording")
            
            start_time = time.time()
            
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Recording stopped by user")
        except Exception as e:
            print(f"❌ Recording error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False
            
            # Save to file
            self._save_to_file(filename)
            print(f"💾 Audio saved to {filename}")
            
    def _save_to_file(self, filename: str):
        """Save recorded audio to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        
    def __del__(self):
        """Cleanup PyAudio"""
        if hasattr(self, 'p'):
            self.p.terminate()

class StrokeSegmenter:
    """Segments audio into individual strokes using advanced onset detection"""
    
    def __init__(self, min_stroke_interval=0.1, onset_threshold=0.3):
        self.min_stroke_interval = min_stroke_interval
        self.onset_threshold = onset_threshold
        
    def segment_strokes(self, audio_file: str, visualize: bool = False) -> List[Tuple[float, float, np.ndarray]]:
        """
        Segment audio into individual strokes
        
        Args:
            audio_file: Path to audio file
            visualize: Whether to show visualization
            
        Returns:
            List of (start_time, end_time, audio_segment) tuples
        """
        print(f"📄 Loading audio file: {audio_file}")
        
        # Load audio using training pipeline function
        audio, sr = get_audio(Path(audio_file))
        print(f"📊 Audio duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")
        
        # Find all onsets in the audio
        onsets = self._find_all_onsets(audio, sr)
        print(f"🎯 Found {len(onsets)} potential onsets")
        
        # Filter onsets based on minimum interval
        filtered_onsets = self._filter_onsets(onsets)
        print(f"✅ Filtered to {len(filtered_onsets)} valid onsets")
        
        # Create segments around each onset
        segments = []
        for i, onset in enumerate(filtered_onsets):
            # Calculate segment boundaries
            start_time = max(0, onset - 0.05)  # 50ms before onset
            end_time = min(len(audio)/sr, onset + 0.15)  # 150ms after onset
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract audio segment
            segment = audio[start_sample:end_sample]
            
            segments.append((start_time, end_time, segment))
            
        if visualize:
            self._visualize_segmentation(audio, sr, filtered_onsets, segments)
            
        return segments
    
    def _find_all_onsets(self, audio: np.ndarray, sr: float) -> np.ndarray:
        """Find all onsets in audio using multiple methods"""
        
        # Method 1: Spectral flux
        spectral_onsets = librosa.onset.onset_detect(
            y=audio, sr=sr, units='time',
            onset_envelope=librosa.onset.onset_strength(y=audio, sr=sr),
            pre_max=3, post_max=3, pre_avg=3, post_avg=3,
            delta=self.onset_threshold, wait=5
        )
        
        # Method 2: Energy-based detection
        rms_features = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        rms_diff = np.diff(rms_features, prepend=rms_features[0])
        rms_diff = np.maximum(0, rms_diff)
        
        # Convert RMS diff to time-based onset detection
        hop_length = 256
        frame_times = librosa.frames_to_time(np.arange(len(rms_diff)), sr=sr, hop_length=hop_length)
        
        # Find peaks in RMS difference
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms_diff, height=np.std(rms_diff) * 1.5, distance=5)
            energy_onsets = frame_times[peaks]
        except ImportError:
            # Simple peak detection if scipy is not available
            energy_onsets = []
            threshold = np.std(rms_diff) * 1.5
            for i in range(1, len(rms_diff) - 1):
                if rms_diff[i] > threshold and rms_diff[i] > rms_diff[i-1] and rms_diff[i] > rms_diff[i+1]:
                    energy_onsets.append(frame_times[i])
            energy_onsets = np.array(energy_onsets)
        
        # Combine all onsets
        all_onsets = np.concatenate([spectral_onsets, energy_onsets])
        
        # Remove duplicates and sort
        if len(all_onsets) > 0:
            all_onsets = np.sort(all_onsets)
        
        return all_onsets
    
    def _filter_onsets(self, onsets: np.ndarray) -> np.ndarray:
        """Filter onsets to remove those too close together"""
        if len(onsets) <= 1:
            return onsets
            
        filtered = [onsets[0]]
        for onset in onsets[1:]:
            if onset - filtered[-1] >= self.min_stroke_interval:
                filtered.append(onset)
                
        return np.array(filtered)
    
    def _visualize_segmentation(self, audio: np.ndarray, sr: float, onsets: np.ndarray, segments: List[Tuple[float, float, np.ndarray]]):
        """Visualize the segmentation results"""
        plt.figure(figsize=(15, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        time_axis = np.linspace(0, len(audio)/sr, len(audio))
        plt.plot(time_axis, audio, alpha=0.7, color='blue')
        
        # Mark onsets
        for onset in onsets:
            plt.axvline(x=onset, color='red', linestyle='--', alpha=0.8)
        
        # Mark segments
        for i, (start, end, _) in enumerate(segments):
            plt.axvspan(start, end, alpha=0.2, color='green')
            plt.text(start + (end-start)/2, max(audio) * 0.8, f'{i+1}', 
                    ha='center', va='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
        plt.title(f'Audio Segmentation - {len(segments)} strokes detected')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot onset strength
        plt.subplot(3, 1, 2)
        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
        times_onset = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=sr)
        plt.plot(times_onset, onset_envelope, color='orange')
        
        for onset in onsets:
            plt.axvline(x=onset, color='red', linestyle='--', alpha=0.8)
            
        plt.title('Onset Strength')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        plt.grid(True, alpha=0.3)
        
        # Plot spectrogram
        plt.subplot(3, 1, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, cmap='viridis')
        
        for onset in onsets:
            plt.axvline(x=onset, color='red', linestyle='--', alpha=0.8)
            
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        plt.tight_layout()
        plt.show()

class StrokeClassifier:
    """Classifies segmented strokes using the trained CNN model"""
    
    def __init__(self, model_path: str, mel_stats_path: Optional[str] = None, 
                 class_names: Optional[List[str]] = None):
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load mel statistics for normalization
        self.mel_stats = self._load_mel_stats(mel_stats_path)
        
        # Class names
        if class_names is None:
            self.class_names = ['Ta', 'Tha', 'Dha', 'Na', 'Thi', 'Ki', 'Dhi', 'Tam', 'Dhom', 'Kita']
        else:
            self.class_names = class_names
            
        print(f"✅ Model loaded successfully")
        print(f"📋 Classes: {self.class_names}")
    
    def _load_model(self, model_path: str) -> MridangamCNN:
        """Load the trained model"""
        try:
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
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
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
    
    def classify_segments(self, segments: List[Tuple[float, float, np.ndarray]], 
                         sr: float = 22050) -> List[Tuple[float, float, str, float, np.ndarray]]:
        """
        Classify each audio segment using the exact training pipeline
        
        Args:
            segments: List of (start_time, end_time, audio_segment) tuples
            sr: Sample rate
            
        Returns:
            List of (start_time, end_time, predicted_class, confidence, probabilities) tuples
        """
        results = []
        
        print(f"🔍 Classifying {len(segments)} segments...")
        
        for i, (start_time, end_time, segment) in enumerate(segments):
            try:
                # Use the exact training preprocessing pipeline
                onset = get_onset(segment, sr)
                audio_window = get_window(onset, segment, sr)
                mel_spec = get_mel_spectrogram(audio_window, sr)
                
                # Normalize and format exactly like training
                mel_tensor = self._normalize_and_format(mel_spec)
                
                # Model inference
                with torch.no_grad():
                    outputs = self.model(mel_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    predicted_class = self.class_names[predicted_idx.item()]
                    confidence_val = confidence.item()
                    all_probs = probabilities.cpu().numpy().flatten()
                    
                    results.append((start_time, end_time, predicted_class, confidence_val, all_probs))
                    
                    print(f"  Segment {i+1}: {predicted_class} (confidence: {confidence_val:.3f})")
                    
            except Exception as e:
                print(f"❌ Error processing segment {i+1}: {e}")
                results.append((start_time, end_time, "Unknown", 0.0, np.zeros(len(self.class_names))))
                
        return results
    
    def _normalize_and_format(self, mel_spec: np.ndarray, target_length: int = 128) -> torch.Tensor:
        """Normalize per mel-bin and format for CNN architecture - exactly like training"""
        # Ensure consistent time dimension
        n_mels, time_frames = mel_spec.shape
        
        # Pad or truncate to target length
        if time_frames < target_length:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, target_length - time_frames)), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_length]
        
        # Per-frequency normalization - exactly like training
        if self.mel_stats is not None:
            # Normalize each mel bin independently
            mel_spec = (mel_spec - self.mel_stats['mean'][:, np.newaxis]) / (self.mel_stats['std'][:, np.newaxis] + 1e-8)
        
        # Convert to tensor and add batch and channel dimensions
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_frames)
        
        return mel_tensor.to(self.device)

class AudioRecordingInferencePipeline:
    """Complete pipeline for audio recording and stroke inference"""
    
    def __init__(self, model_path: str, mel_stats_path: Optional[str] = None,
                 class_names: Optional[List[str]] = None, confidence_threshold: float = 0.5):
        
        self.recorder = AudioRecorder()
        self.segmenter = StrokeSegmenter()
        self.classifier = StrokeClassifier(model_path, mel_stats_path, class_names)
        self.confidence_threshold = confidence_threshold
        
    def run_full_pipeline(self, output_dir: str = "recordings", 
                         recording_duration: Optional[float] = None,
                         visualize: bool = False) -> Dict[str, Any]:
        """
        Run the complete pipeline: Record -> Segment -> Classify
        
        Args:
            output_dir: Directory to save recordings
            recording_duration: Duration to record (None for manual stop)
            visualize: Whether to show segmentation visualization
            
        Returns:
            Dictionary with results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
        
        print("🎵 Starting Audio Recording and Inference Pipeline")
        print("=" * 60)
        
        # Step 1: Record audio
        print("\n📹 STEP 1: Recording Audio")
        self.recorder.start_recording(audio_filename, recording_duration)
        
        # Step 2: Segment strokes
        print("\n✂️ STEP 2: Segmenting Strokes")
        segments = self.segmenter.segment_strokes(audio_filename, visualize=visualize)
        
        if len(segments) == 0:
            print("⚠️ No strokes detected in the recording")
            return {"audio_file": audio_filename, "segments": [], "results": []}
        
        # Step 3: Classify segments
        print("\n🔍 STEP 3: Classifying Strokes")
        results = self.classifier.classify_segments(segments)
        
        # Step 4: Filter and summarize results
        print("\n📊 STEP 4: Results Summary")
        filtered_results = [(start, end, class_name, conf, probs) 
                          for start, end, class_name, conf, probs in results 
                          if conf >= self.confidence_threshold]
        
        print("=" * 60)
        print(f"📄 Audio file: {audio_filename}")
        print(f"🎯 Total segments: {len(segments)}")
        print(f"✅ High-confidence detections: {len(filtered_results)}")
        print(f"🎚️ Confidence threshold: {self.confidence_threshold}")
        
        class_counts = {}
        if len(filtered_results) > 0:
            print("\n🥁 DETECTED STROKES:")
            print("-" * 50)
            for i, (start, end, class_name, conf, probs) in enumerate(filtered_results):
                print(f"{i+1:2d}. {start:6.2f}s - {end:6.2f}s: {class_name:>6} (conf: {conf:.3f})")
                
                # Show top 3 predictions for high confidence
                if conf > 0.7:
                    top_indices = np.argsort(probs)[::-1][:3]
                    top_preds = [(self.classifier.class_names[idx], probs[idx]) for idx in top_indices]
                    print(f"     Top 3: {' | '.join([f'{name}: {prob:.2f}' for name, prob in top_preds])}")
            
            # Class distribution
            for _, _, class_name, _, _ in filtered_results:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\n📈 CLASS DISTRIBUTION:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(filtered_results)) * 100
                print(f"  {class_name:>6}: {count:2d} ({percentage:.1f}%)")
                
        else:
            print("⚠️ No high-confidence detections found")
            print("💡 Try adjusting the confidence threshold or check audio quality")
        
        return {
            "audio_file": audio_filename,
            "segments": segments,
            "results": results,
            "filtered_results": filtered_results,
            "class_counts": class_counts
        }

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Recording and Stroke Inference Pipeline")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--mel-stats", "-s", help="Path to mel statistics file")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--duration", "-d", type=float, help="Recording duration in seconds")
    parser.add_argument("--output-dir", "-o", default="recordings", help="Output directory for recordings")
    parser.add_argument("--visualize", "-v", action="store_true", help="Show segmentation visualization")
    parser.add_argument("--classes", help="Comma-separated list of class names")
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = None
    if args.classes:
        class_names = [name.strip() for name in args.classes.split(',')]
    
    # Create pipeline
    pipeline = AudioRecordingInferencePipeline(
        model_path=args.model,
        mel_stats_path=args.mel_stats,
        class_names=class_names,
        confidence_threshold=args.confidence
    )
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            output_dir=args.output_dir,
            recording_duration=args.duration,
            visualize=args.visualize
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {results['audio_file']}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 