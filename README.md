# 🩺 Medical Chatbot LLM

An AI-powered medical assistant capable of analyzing medical images and answering health-related queries.  
The chatbot is built on **Large Language Models (LLMs)** with integration of image understanding to identify possible medical conditions based on uploaded images.  
It provides **explanations with reasoning**, supports **multi-modal inputs** (text + images), and can detect potential issues like skin cancer or chest anomalies.

---

## 📌 Features

- 🧠 **Medical Q&A** – Ask any medical question and get contextually relevant, evidence-based responses.
- 🩻 **Image Analysis** – Upload medical images (X-rays, skin lesion photos, etc.) for AI-powered analysis.
- 🔍 **Condition Detection** – Identifies possible symptoms or abnormalities with explanations.
- 💬 **Conversational Interface** – Natural, human-like interaction for non-technical users.
- 📊 **Explainable Results** – Includes reasoning and possible causes for predictions.

---

## 📂 Project Structure

Medical-Chatbot-LLM/
├── app.py # Main Streamlit application
├── model/ # Trained model files
├── utils/ # Helper functions for image/text processing
├── Demo Images/ # Example input images for testing
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 How It Works

1. **Upload an image** (X-ray, skin lesion, etc.) or type a medical query.
2. The **LLM model** processes the text and/or image.
3. If an image is provided, the vision module extracts features and detects possible abnormalities.
4. The chatbot generates a **diagnosis suggestion with reasoning**.
5. The final output is displayed in an **interactive chat interface**.

---

## 🖼 Demo

**Skin Lesion Detection**  
<img src="Demo Images/Screenshot_1.png" width="500"/>

**Chest X-ray Analysis**  
<img src="Demo Images/Screenshot_2.png" width="500"/>

**Multiple Image Checks**  
<img src="Demo Images/Screenshot_3.png" width="500"/>

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – Web app interface
- **OpenAI GPT / LLaVA / BLIP** – LLM + image understanding
- **PyTorch** – Model training & inference
- **Pillow / OpenCV** – Image preprocessing

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Ridit07/Medical-Chatbot-LLM.git
cd Medical-Chatbot-LLM

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```


