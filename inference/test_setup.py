#!/usr/bin/env python3
"""
Test script to validate the real-time inference setup
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import librosa
        print(f"✅ Librosa: {librosa.__version__}")
    except ImportError:
        print("❌ Librosa not available")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import pyaudio
        print("✅ PyAudio available")
    except ImportError:
        print("⚠️  PyAudio not available - real-time audio capture won't work")
        print("   Install with: pip install pyaudio")
    
    return True

def test_model_loading():
    """Test if we can load the CNN with attention model"""
    print("\n🧠 Testing model loading...")
    
    try:
        from realtime_cnn_attention_inference import MridangamCNN
        
        # Create a test model
        model = MridangamCNN(n_mels=128, num_classes=10, dropout_rate=0.5)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        import torch
        test_input = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("\n🔄 Testing preprocessing pipeline...")
    
    try:
        import numpy as np
        from realtime_cnn_attention_inference import get_onset_local, get_window_local, get_mel_spectrogram_local
        
        # Create dummy audio data
        sr = 22050
        duration = 1.0  # 1 second
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        # Test onset detection
        onset = get_onset_local(audio, sr)
        print(f"✅ Onset detection: {onset}")
        
        # Test window extraction
        window = get_window_local(onset, audio, sr)
        print(f"✅ Window extraction: {window.shape}")
        
        # Test mel spectrogram
        mel_spec = get_mel_spectrogram_local(window, sr)
        print(f"✅ Mel spectrogram: {mel_spec.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

def find_available_models():
    """Find available model checkpoints"""
    print("\n📁 Looking for available models...")
    
    # Check different possible locations
    locations = [
        Path("../model_checkpoints"),
        Path("model_weights"),
        Path("../inference/model_weights")
    ]
    
    found_models = []
    
    for location in locations:
        if location.exists():
            models = list(location.glob("*.pth"))
            if models:
                print(f"✅ Found {len(models)} model(s) in {location}:")
                for model in models:
                    print(f"   - {model.name} ({model.stat().st_size / 1024 / 1024:.1f} MB)")
                    found_models.append(model)
    
    if not found_models:
        print("⚠️  No model checkpoints found")
        print("   Please ensure you have trained models in:")
        for loc in locations:
            print(f"   - {loc}")
    
    return found_models

def test_audio_devices():
    """Test available audio input devices"""
    print("\n🎤 Testing audio devices...")
    
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        print("Available audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"   {i}: {info['name']} (SR: {int(info['defaultSampleRate'])})")
        
        p.terminate()
        print("✅ Audio devices accessible")
        return True
        
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Real-Time Mridangam Inference Setup Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Dependencies
    if test_imports():
        tests_passed += 1
    
    # Test 2: Model loading
    if test_model_loading():
        tests_passed += 1
    
    # Test 3: Preprocessing
    if test_preprocessing():
        tests_passed += 1
    
    # Test 4: Audio devices
    if test_audio_devices():
        tests_passed += 1
    
    # Find models
    models = find_available_models()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Models found: {len(models)}")
    
    if tests_passed == total_tests and models:
        print("\n🎉 Setup is ready for real-time inference!")
        print("\nNext steps:")
        print("1. Run example_realtime_usage.py for interactive examples")
        print("2. Or use realtime_cnn_attention_inference.py directly")
        
        if models:
            latest_model = max(models, key=lambda p: p.stat().st_mtime)
            print(f"\nRecommended model: {latest_model}")
            print(f"Command: python realtime_cnn_attention_inference.py --model \"{latest_model}\"")
    else:
        print("\n⚠️  Setup incomplete. Please resolve the issues above.")
        
        if not models:
            print("\n💡 To train a model, run the training script in model_architecture/")

if __name__ == "__main__":
    main() 