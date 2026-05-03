# Skill Learning AI: Vision-Powered Action Analyzer

<a href="https://huggingface.co/spaces/Ronxak/Skill-Learning-AI">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge" alt="Live Demo on Hugging Face">
  </a>
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO11-Vision-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama-black?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge)

**Skill Learning AI** is an innovative computer vision and natural language processing pipeline designed to watch video demonstrations and automatically deduce actionable, step-by-step textual instructions. By combining the blazing-fast object detection capabilities of YOLO11 with the advanced temporal reasoning of large language models via Groq, this application serves as an automated "Reverse Engineer" for human activities.

Whether you're watching a cooking tutorial, a mechanical repair, a DIY craft, or a workout routine, Skill Learning AI breaks down the continuous stream of video into a structured, logical set of tasks.

## 🎯 Use Cases

- **Automated SOP Generation:** Watch factory workers or technicians perform tasks and automatically generate Standard Operating Procedures (SOPs).
- **Recipe Extraction:** Upload a cooking video and get a step-by-step recipe based on the ingredients and utensils shown.
- **Fitness & Training:** Analyze workout videos to break down complex movements into sequential phases.
- **Educational Accessibility:** Convert visually complex video tutorials into screen-reader-friendly text guides.

## 🔬 How It Works (Deep Dive)

Rather than relying on massive, slow, and expensive Video-LLMs, this project uses a highly efficient, modular, and interpretable pipeline. 

### Phase 1: Temporal Sub-Sampling (OpenCV)
Videos contain thousands of frames, many of which are highly redundant. `pipeline.py` uses OpenCV to intelligently sub-sample the video, extracting key frames at regular intervals (e.g., 1 frame every 2 seconds). This drastically reduces the computational load while preserving the narrative arc of the action.

### Phase 2: Spatial Object Recognition (YOLO11)
`detector.py` processes the extracted frames through Ultralytics YOLO11 Nano. Instead of just identifying objects, it builds a chronological "timeline" of the video. It filters out low-confidence detections and maps exactly which objects appear at which timestamp. This transforms heavy pixel data into lightweight, structured JSON data (`frame_timeline.json`).

### Phase 3: Cognitive Temporal Reasoning (Groq LLM)
A raw list of objects over time is meaningless without context. `reasoner.py` feeds this JSON timeline to an advanced Llama instruction model via the Groq API. The LLM acts as the cognitive engine: it filters out visual noise, identifies the overarching intent of the human actor, and translates the object timeline into human-readable, chronological instructions.

## 🏗️ Project Architecture

```text
[ Uploaded Video ] 
        │
        ▼
┌───────────────────┐
│ 1. Frame Extractor│ (OpenCV) -> Extracts 1 frame / 2 sec
└───────────────────┘
        │
        ▼ [ Images ]
┌───────────────────┐
│ 2. YOLO Vision    │ (YOLO11n) -> Detects objects & builds timeline
└───────────────────┘
        │
        ▼ [ JSON Data: { "Frame_1": ["bowl", "apple"], ... } ]
┌───────────────────┐
│ 3. Groq Reasoner  │ (Meta-Llama) -> Deduces steps & intent
└───────────────────┘
        │
        ▼
[ Step-by-Step Instructions Output ]
```

## ✨ Core Features

- **Blazing Fast Performance:** By utilizing the Nano version of YOLO11 and Groq's high-speed inference engine, processing is incredibly fast.
- **Interpretable Data Bridge:** The intermediary JSON timeline allows developers to see exactly *what* the AI saw before the LLM makes its deductions.
- **Noise Resilient:** The prompting strategy explicitly instructs the LLM to ignore random detection anomalies (e.g., a misclassified object flashing for one frame).
- **Interactive UI:** Built entirely in Gradio, providing a clean, accessible web interface for easy drag-and-drop video analysis.

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.8+
- A [Groq API Key](https://console.groq.com/keys)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/skill-learning-ai.git
   cd skill-learning-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the App

Start the full Gradio pipeline:
```bash
python app.py
```
*Note: The YOLO11n model will automatically download on the first run.*

## 📈 Example Output

**Input:** A video of someone making a sandwich.

**AI Output:**
> **Overall Action:** Making a sandwich.
> 
> **Steps:**
> 1. **Preparation:** Gathering bread and a knife.
>    <br>*Explanation:* Detected 'bread' and 'knife' in the initial frames.
> 2. **Adding Ingredients:** Applying a spread and adding vegetables.
>    <br>*Explanation:* 'bowl' (likely containing spread) and 'broccoli' (proxy for vegetables) appeared in the mid-frames.
> 3. **Finalizing:** Closing the sandwich.
>    <br>*Explanation:* Hands moving over the completed stack in the final frames.

## License

This project is licensed under the MIT License.

---
*Built with utilizing Ultralytics, Groq, and Gradio.*
