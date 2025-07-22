Gesture-Controlled Puzzle Game 🖐️✨
This project is an interactive puzzle game that lets you control on-screen objects using hand gestures. Through your webcam, it tracks your hand movements in real-time, allowing you to pinch, grab, and move shapes to solve the puzzle.

The game is built with Python, using OpenCV for video processing and Google's MediaPipe for high-fidelity hand landmark detection.

Features
Real-Time Gesture Control: Uses a standard webcam to track your hand and fingers.

Intuitive Pinch Mechanic: Easily grab and drag objects by pinching your thumb and fingers together.

Visual Feedback: Overlays a skeletal model of your hand for clear interaction.

Smooth Cursor Movement: Implements smoothing to filter out jitter and ensure fluid object control.

Replayable Puzzles: The starting positions of shapes are randomized for a new challenge every time you play.

Immersive UI: The entire game is rendered as an overlay on your live webcam feed.

Tech Stack
Python 3

OpenCV: For capturing and processing the webcam feed.

MediaPipe: For robust hand and finger tracking.

NumPy: For numerical operations and shape manipulation.
