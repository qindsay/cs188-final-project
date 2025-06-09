# CS188 Final Project
This repository is the Spring 2025 final project for UCLA's CS188 (Intro to Robotics) course taught by Professor Yuchen Cui. 

## Project Overview
This is the default project, which completes the Square Assembly task from Robosuite. The robot observes multiple demonstrations and learns to reproduce the behavior required to complete the task successfully. We used advanced DMPs to complete this task. 

## Repository Structure
```
cs188-final-project/
├── __pycache__/                  # Python cache files
├── demos.npz                     # Demonstration data
│
├── dmp.py                        # Core DMP logic
├── dmp_policy.py                # Main DMP-based control policy
├── dmp_policy_previous.py       # Old version of the policy
│
├── load_data.py                 # Functions to load and preprocess demo data
├── multiple_demos.py            # Logic for selecting/handling multiple demos
├── pid.py                       # PID controller implementation
├── test_ca3.py                  # Testing script (adapted from CA3)
│
├── README.md
```

## How to Run:
1. Make sure you have already installed robosuite before running.
   ```pip install robosuite```
2. Clone the repository:
    ```bash
    git clone https://github.com/qindsay/cs188-final-project.git
    cd cs188-final-project
3. Run the code: ```mjpython test_ca3.py```

## Contributors:
Evelyn Cho, Lindsay Qin
