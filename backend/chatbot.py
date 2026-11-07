import random
import requests
import os

# Large knowledge base (expanded)
KNOWLEDGE = {
    "what is deepfake": [
        "A deepfake is synthetic media created using artificial intelligence, where a person's likeness is replaced with someone else's. It often uses generative adversarial networks (GANs).",
        "Deepfakes are AI-generated videos or images that manipulate a person's appearance or voice. They can be used for entertainment, but also for misinformation and fraud."
    ],
    "how to detect deepfake": [
        "Look for unnatural blinking, inconsistent lighting, unusual mouth movements, and artefacts around face edges. Use tools like this Deepfake Detector!",
        "Detection methods include analyzing facial landmarks, skin texture, eye reflections, and temporal inconsistencies across frames."
    ],
    "ethics": [
        "Deepfakes raise serious ethical concerns: they can be used for identity theft, political disinformation, and non‑consensual pornography. Responsible use and detection are crucial.",
        "Ethical guidelines include mandatory watermarking of AI‑generated content, media literacy education, and legal frameworks to penalize malicious use."
    ],
    "future of deepfake": [
        "Future detection will involve multi‑modal analysis (visual + audio + metadata), frequency domain analysis, GAN fingerprints, and blockchain provenance tracking.",
        "As deepfakes improve, detection tools will also evolve using more advanced neural networks and larger datasets."
    ],
    "what can i do": [
        "You can upload a video or image to our detector. You can also educate others about deepfakes and support research in AI forensics.",
        "Use our tool regularly, share your results, and stay updated on the latest detection techniques."
    ],
    "how accurate": [
        "Our model achieves over 90% accuracy on the SDFVD dataset, with precision and recall around 85‑95%. Metrics are displayed on the upload page.",
        "The model's performance is continuously improving as we add more training data and refine the architecture."
    ]
}

def chatbot_response(query):
    """
    Return a response based on keyword matching.
    If no match, ask if the user wants to use the AI API.
    """
    query_lower = query.lower()
    for key, responses in KNOWLEDGE.items():
        if key in query_lower:
            return random.choice(responses)
    return None

def get_ai_response(query):
    """
    Use Hugging Face Inference API (free) for a more intelligent answer.
    Requires HF_TOKEN environment variable.
    """
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        return None

    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": f"Answer the following question about deepfakes concisely: {query}",
        "parameters": {"max_new_tokens": 150, "temperature": 0.7}
    }
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                return data[0].get("generated_text", "").strip()
    except:
        pass
    return None