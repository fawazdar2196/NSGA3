# 🌱 NSGA-III: Multi-Objective Evolutionary Algorithm

This repository contains a Python implementation of the **NSGA-III (Non-dominated Sorting Genetic Algorithm III)**, a state-of-the-art evolutionary algorithm designed for solving **multi-objective optimization problems**.  

The implementation includes modules for selection, crossover, mutation, reference point generation, and performance evaluation using standard metrics.

---

## 📌 Overview
- **NSGA-III** extends the well-known NSGA-II algorithm by introducing **reference points** for better diversity preservation in many-objective optimization.  
- The algorithm is particularly effective when dealing with problems with **more than three objectives**.  
- This project provides a modular and extensible Python framework to test NSGA-III on different benchmark functions or real-world optimization tasks.  

---

## ✨ Features
- ✅ Non-dominated sorting  
- ✅ Reference point generation and association  
- ✅ Crossover & mutation operators  
- ✅ Population normalization and scalarization  
- ✅ Ideal point update during evolution  
- ✅ Hypervolume (HV) calculation for performance measurement  
- ✅ Easily customizable for different optimization problems  

---

## 🛠️ Tech Stack
- **Language:** Python 3  
- **Core Libraries:**  
  - `numpy` – numerical computations  
  - `matplotlib` – (optional) visualization of Pareto fronts  

---


## 🚀 Installation & Setup
```bash
# 1) Clone the repository
git clone https://github.com/fawazdar2196/nsga-iii.git
cd nsga-iii

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the algorithm
python nsga3.py
