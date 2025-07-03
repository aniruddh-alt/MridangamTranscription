#!/usr/bin/env python3
"""
Example usage of the Audio Recording Inference Pipeline

This script demonstrates how to use the new pipeline for mridangam stroke detection:
1. Record audio from microphone
2. Segment strokes automatically
3. Classify each stroke using the trained model
4. Display results with confidence scores

Usage:
    python example_recording_inference.py --model path/to/model.pth --duration 10
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from inference.audio_recording_inference import AudioRecordingInferencePipeline

def main():
    """Example of using the audio recording inference pipeline"""
    
    parser = argparse.ArgumentParser(description="Example Audio Recording Inference")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--mel-stats", "-s", help="Path to mel statistics file")
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--visualize", "-v", action="store_true", help="Show segmentation visualization")
    
    args = parser.parse_args()
    
    print("🎵 Mridangam Stroke Detection - Recording Example")
    print("=" * 60)
    
    # Create the pipeline
    pipeline = AudioRecordingInferencePipeline(
        model_path=args.model,
        mel_stats_path=args.mel_stats,
        confidence_threshold=args.confidence
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Mel Stats: {args.mel_stats or 'None (using default normalization)'}")
    print(f"  Duration: {args.duration}s")
    print(f"  Confidence Threshold: {args.confidence}")
    print(f"  Visualization: {args.visualize}")
    
    print(f"\n🎤 Get ready to play mridangam strokes!")
    print(f"⏱️ Recording will start in 3 seconds...")
    
    import time
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print("🔴 Recording started!")
    
    # Run the full pipeline
    try:
        results = pipeline.run_full_pipeline(
            output_dir="example_recordings",
            recording_duration=args.duration,
            visualize=args.visualize
        )
        
        print(f"\n✅ Recording and analysis complete!")
        print(f"📁 Audio saved to: {results['audio_file']}")
        
        if len(results['filtered_results']) > 0:
            print(f"\n🎯 Summary:")
            print(f"  Total strokes detected: {len(results['filtered_results'])}")
            print(f"  Most common stroke: {max(results['class_counts'].items(), key=lambda x: x[1])[0]}")
            
            # Show stroke sequence
            stroke_sequence = [stroke for _, _, stroke, _, _ in results['filtered_results']]
            print(f"  Stroke sequence: {' -> '.join(stroke_sequence)}")
            
        else:
            print(f"\n⚠️ No strokes detected with confidence >= {args.confidence}")
            print(f"💡 Try:")
            print(f"  - Playing strokes closer to the microphone")
            print(f"  - Reducing the confidence threshold (--confidence 0.3)")
            print(f"  - Checking microphone permissions")
    
    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 