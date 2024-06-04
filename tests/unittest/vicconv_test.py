import unittest
import numpy as np
from Vid2Frames import vid2frames

class TestProcessVideo(unittest.TestCase):
    def test_process_video(self):
        # Create a sample video
        video_frames = np.random.rand(10, 224, 224, 3)  # 10 frames, 224x224 pixels, 3 color channels

        # Process the video
        processed_video = vid2frames(video_frames)

        # Check the shape of the output
        self.assertEqual(processed_video.shape, (10, 224, 224, 3))

        # Check that at least one frame from each second is included
        for i in range(0, 10, 2):
            self.assertTrue(np.any(processed_video[i]))

        # Check that the output can be directly fed to a CNN
        # This might involve checking the data type or other properties
        self.assertEqual(processed_video.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
