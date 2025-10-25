# Hand Tracking Cursor Control 🖐️🖱️

Turn your hand into a **virtual mouse**! This Python project uses **MediaPipe** for hand tracking and **PyAutoGUI** for controlling the mouse. Move your index finger to move the cursor and perform clicks by pinching your thumb and index finger.

---

## Features

- Smooth cursor movement using hand tracking  
- Click gesture detection (thumb + index pinch)  
- Visual feedback: green circle on the index finger and "CLICKED!" text when a click is detected  
- Works with any standard webcam  

---

## Demo

🎥 Suggested preview:  

1. Show the hand in front of the camera with a **green circle** highlighting the index finger  
2. Move your hand → cursor moves smoothly  
3. Pinch thumb + index → click occurs (show “CLICKED!” feedback)  
4. Optional: click a browser tab, folder, or video to demonstrate interaction  

---

## Screenshots

![Hand Tracking Cursor](screenshots/demo.png)  
*Green circle highlights index finger; "CLICKED!" appears on click.*

---

## Installation

```bash
git clone https://github.com/yourusername/hand-tracking-cursor.git
cd hand-tracking-cursor
```

---

## Requirements

```bash
pip install -r requirements.txt

