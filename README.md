# MiguelAnhalt-machine_learning_ai_seminar_task_7

## Setup Instructions

### 1. Install WSL (if not already installed)
```
wsl --install
```

### 2. Update and install Python, venv, and pip
```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

### 3. (Optional but recommended) Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### 4. Install system dependencies for matplotlib/Qt support (for plotting)
```
sudo apt install -y libqt5gui5 libqt5widgets5 libqt5core5a libxcb-xinerama0 python3-tk
```

### 5. Install Python dependencies
```
pip install -r requirements.txt
```

---

## Python dependencies
See `requirements.txt` for the list of required Python packages.

---

## Running the script

After completing the above steps, you can run the example script:
```
python Session7_Example_fashion_mnist.py
```
