#!/usr/bin/env python3
"""
Performance test script for different tracking algorithms
"""
import cv2
import time
import numpy as np
from src.tracker import ImageTracker

def test_tracker_performance(tracker_type, test_duration=10):
    """Test tracking performance for a specific tracker type"""
    print(f"\n=== Testing {tracker_type} Tracker ===")
    
    # Create a simple test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a simple object to track (white rectangle)
    cv2.rectangle(frame, (250, 200), (350, 300), (255, 255, 255), -1)
    
    # Initialize tracker
    tracker = ImageTracker(tracker_type=tracker_type)
    bbox = (250, 200, 100, 100)  # x, y, w, h
    
    success = tracker.initialize(frame, bbox)
    if not success:
        print(f"Failed to initialize {tracker_type} tracker")
        return None
    
    # Test tracking performance
    times = []
    iterations = 0
    start_time = time.time()
    
    while time.time() - start_time < test_duration:
        # Simulate slight movement
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        offset_x = int(5 * np.sin(iterations * 0.1))
        offset_y = int(5 * np.cos(iterations * 0.1))
        cv2.rectangle(frame, 
                     (250 + offset_x, 200 + offset_y), 
                     (350 + offset_x, 300 + offset_y), 
                     (255, 255, 255), -1)
        
        # Time the tracking update
        track_start = time.time()
        success, bbox = tracker.update(frame)
        track_time = (time.time() - track_start) * 1000  # ms
        
        if success:
            times.append(track_time)
        iterations += 1
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps_estimate = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"Iterations: {len(times)}")
        print(f"Average time: {avg_time:.2f}ms")
        print(f"Min time: {min_time:.2f}ms")
        print(f"Max time: {max_time:.2f}ms")
        print(f"Estimated FPS: {fps_estimate:.1f}")
        
        return {
            'tracker': tracker_type,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps_estimate': fps_estimate,
            'iterations': len(times)
        }
    else:
        print("No successful tracking updates")
        return None

def main():
    """Run performance tests for all trackers"""
    # Test legacy trackers that should be fast
    trackers = ['MOSSE', 'KCF', 'CSRT', 'MIL']
    results = []
    
    print("OpenCV Tracker Performance Test")
    print("=" * 40)
    
    for tracker_type in trackers:
        try:
            result = test_tracker_performance(tracker_type, test_duration=5)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error testing {tracker_type}: {e}")
    
    # Summary
    print("\n" + "=" * 40)
    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    
    if results:
        # Sort by speed (lowest time = fastest)
        results.sort(key=lambda x: x['avg_time'])
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['tracker']}: {result['avg_time']:.2f}ms avg ({result['fps_estimate']:.1f} FPS)")
        
        # Recommendation
        fastest = results[0]
        print(f"\n🚀 RECOMMENDATION: Use '{fastest['tracker']}' for best performance!")
    else:
        print("No successful tracker tests completed")

if __name__ == "__main__":
    main() 