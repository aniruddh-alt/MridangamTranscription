#!/usr/bin/env python3
"""
Example usage of the Real-Time Mridangam CNN with Attention Inference Pipeline

This script demonstrates how to use the real-time inference system with your trained models.
"""

from pathlib import Path
import os
import sys
from realtime_cnn_attention_inference import run_realtime_detection, RealTimeStrokeDetector

def find_latest_model():
    """Find the latest trained model checkpoint"""
    model_dir = Path("../model_checkpoints")
    
    # Look for best model files first
    best_models = list(model_dir.glob("best_model_*.pth"))
    if best_models:
        # Sort by creation time and return the latest
        latest_model = max(best_models, key=lambda p: p.stat().st_mtime)
        return latest_model
    
    # Fall back to other model locations
    inference_models = Path("model_weights")
    if inference_models.exists():
        model_files = list(inference_models.glob("*.pth"))
        if model_files:
            return model_files[0]
    
    return None

def example_basic_detection():
    """Basic real-time detection example"""
    print("🔍 Looking for trained model...")
    
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found!")
        print("Please ensure you have a trained model in:")
        print("  - ../model_checkpoints/best_model_*.pth")
        print("  - model_weights/*.pth")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Run basic detection for 30 seconds
    print("\n🚀 Starting basic real-time detection...")
    print("💡 Play your mridangam and watch the detections!")
    
    run_realtime_detection(
        model_path=str(model_path),
        confidence_threshold=0.4,  # Lower threshold to catch more strokes
        duration=30  # Run for 30 seconds
    )

def example_custom_classes():
    """Example with custom class names"""
    print("🔍 Looking for trained model...")
    
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found!")
        return
    
    # Custom mridangam stroke names (adjust based on your training data)
    custom_classes = [
        'Ta', 'Tha', 'Dha', 'Na', 'Thi', 
        'Ki', 'Dhi', 'Tam', 'Dhom', 'Kita'
    ]
    
    print(f"✅ Found model: {model_path}")
    print(f"🏷️ Using custom classes: {custom_classes}")
    
    run_realtime_detection(
        model_path=str(model_path),
        class_names=custom_classes,
        confidence_threshold=0.5,
        duration=60  # Run for 1 minute
    )

def example_high_precision():
    """Example with high precision settings"""
    print("🔍 Looking for trained model...")
    
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found!")
        return
    
    print(f"✅ Found model: {model_path}")
    print("🎯 High precision mode - only high-confidence detections")
    
    run_realtime_detection(
        model_path=str(model_path),
        confidence_threshold=0.8,  # High threshold for precision
        duration=None  # Run indefinitely until Ctrl+C
    )

def example_programmatic_usage():
    """Example of programmatic usage with custom processing"""
    print("🔍 Looking for trained model...")
    
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found!")
        return
    
    print(f"✅ Found model: {model_path}")
    print("🔧 Programmatic usage example...")
    
    # Create detector instance
    detector = RealTimeStrokeDetector(
        model_path=str(model_path),
        confidence_threshold=0.5
    )
    
    # Custom processing of detections
    stroke_sequence = []
    detection_count = 0
    
    try:
        print("🎵 Listening for stroke patterns...")
        print("💡 Try playing some rhythmic patterns!")
        
        for predicted_class, confidence, timestamp, all_probs in detector.run_detection(duration=20):
            detection_count += 1
            stroke_sequence.append(predicted_class)
            
            # Print detection with custom formatting
            print(f"🥁 #{detection_count:2d}: {predicted_class} ({confidence:.2f})")
            
            # Analyze patterns every 5 strokes
            if len(stroke_sequence) >= 5:
                recent_pattern = " → ".join(stroke_sequence[-5:])
                print(f"📊 Recent pattern: {recent_pattern}")
            
            # Look for specific combinations
            if len(stroke_sequence) >= 3:
                last_three = stroke_sequence[-3:]
                if last_three == ['Ta', 'Dha', 'Mi']:  # Example pattern
                    print("🎉 Detected 'Ta-Dha-Mi' pattern!")
    
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped")
    
    # Analysis summary
    print(f"\n📈 Captured {len(stroke_sequence)} strokes")
    if stroke_sequence:
        print(f"🎵 Full sequence: {' → '.join(stroke_sequence)}")
        
        # Count unique strokes
        from collections import Counter
        stroke_counts = Counter(stroke_sequence)
        print("📊 Stroke distribution:")
        for stroke, count in stroke_counts.most_common():
            print(f"  {stroke}: {count} times")

def main():
    """Main menu for examples"""
    print("🥁 Real-Time Mridangam Stroke Detection Examples")
    print("=" * 50)
    print("Choose an example to run:")
    print("1. Basic real-time detection (30 seconds)")
    print("2. Custom class names (60 seconds)")  
    print("3. High precision mode (until Ctrl+C)")
    print("4. Programmatic usage with pattern analysis")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                example_basic_detection()
                break
            elif choice == "2":
                example_custom_classes()
                break
            elif choice == "3":
                example_high_precision()
                break
            elif choice == "4":
                example_programmatic_usage()
                break
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main() 