# LearnAI – Personalized Learning Recommendation Web App

## Overview

LearnAI is a **Progressive Web App (PWA)** that collects a user’s learning preferences and, via a server‑side AI, returns a **personalized learning plan**.

* **Frontend** – static HTML / CSS / JS, hosted on **GitHub Pages**.  
* **Backend** – Flask API that loads **Qwen/Qwen3‑0.6B**, runs on any cloud provider that can keep a Python process alive (Render, Railway, Fly.io, etc.).  

The two parts communicate over HTTPS:

