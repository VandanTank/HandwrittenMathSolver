# Handwritten Math Solver

A deep learning–based desktop application that recognizes and solves handwritten mathematical expressions drawn on-screen.  
This project demonstrates the integration of **Computer Vision**, **Deep Learning**, and **Graphical User Interfaces (GUI)** in Python.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Model Details](#model-details)
- [Demonstration](#demonstration)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This project enables a user to **draw mathematical equations by hand** (for example, `4 + 5`),  
automatically **recognizes the handwritten symbols**, and **computes the result** in real-time using a trained neural network.

It combines:
- Image preprocessing for handwritten symbol segmentation  
- A CNN model trained on **MNIST digits** and extended math symbols  
- A Tkinter-based GUI for user interaction

---

## Features
- Real-time handwriting recognition through mouse input  
- Supports digits and operators: `+`, `-`, `*`, `/`, `(`, `)`, `.`  
- Automatically evaluates mathematical expressions  
- Model trained on an extended MNIST dataset  
- Simple, lightweight Tkinter GUI  
- Fully offline and open source  

---

## Tech Stack

**Languages & Frameworks:**  
- Python  
- TensorFlow / Keras  
- Tkinter (for GUI)

**Libraries Used:**  
- NumPy  
- PIL (Pillow)  
- SciPy  
- Matplotlib (optional, for visualization)

---

## Project Structure
HandwrittenMathSolve/
│
├── src/
│ ├── model.py # Core symbol detection + CNN loader
│ ├── utils.py # Image preprocessing utilities
│ ├── Train Model.ipynb # CNN training notebook
│ └── Check symbol dataset.ipynb
│
├── calculator.py # Main GUI application
├── requirements.txt # Dependencies
├── .gitignore
└── README.md


---

## Installation & Usage

# Clone the repository
git clone https://github.com/VandanTank/HandwrittenMathSolve.git
cd HandwrittenMathSolve

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python calculator.py

Then, simply **draw your equation** on the canvas window and watch it recognize and solve it in real time.

---

## Model Details

- **Model Type:** Convolutional Neural Network (CNN)  
- **Training Dataset:**  
  - MNIST digits (`0–9`)  
  - Custom symbols for `+`, `-`, `*`, `/`, `(`, `)`, `.`  
- **Input Image Size:** 28×28 grayscale  
- **Framework:** Keras (TensorFlow backend)  
- **Saved Model:** `src/model.h5`

---

## Demonstration

### Example Output:
> **Drawn Expression →** `4 + 5`  
> **Model Output →** `9`


### Demo
![Handwritten Math Solver Demo](demo.gif)

## How It Works

1. **Drawing Capture** – The canvas captures mouse strokes as a grayscale image.  
2. **Image Preprocessing** – The image is thresholded and cleaned using morphological filters.  
3. **Symbol Segmentation** – Connected components are detected to separate digits and operators.  
4. **Prediction** – Each component is resized to 28×28 and classified by the CNN model.  
5. **Expression Evaluation** – Recognized symbols are concatenated and evaluated using Python’s `eval()`.

---

## Future Improvements

- Add support for exponents, fractions, and trigonometric functions.  
- Improve accuracy using a larger, balanced dataset.  
- Deploy the model as a web app using Flask or Streamlit.  
- Add handwriting smoothing for smoother strokes.  
- Integrate GPU inference for real-time speedup.

---

## Author

**Vandan Tank**  
B.Tech in Computer Science and Engineering  
[GitHub Profile](https://github.com/VandanTank)

---

## License
