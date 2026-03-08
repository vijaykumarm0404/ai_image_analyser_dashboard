import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="API_KEY")


def classify_scene(detected_objects):
    labels = [obj["label"] for obj in detected_objects]

    if not labels:
        return "Unknown Scene"

    if any(obj in labels for obj in ["car", "truck", "bus", "motorcycle"]):
        return "Outdoor / Traffic Area"

    if any(obj in labels for obj in ["sofa", "tv", "bed"]):
        return "Home Environment"

    if any(obj in labels for obj in ["laptop", "keyboard", "mouse"]):
        return "Office Environment"

    if "person" in labels and "book" in labels:
        return "Classroom / Study Area"

    if "person" in labels:
        return "Public / General Area"

    return "General Scene"


def generate_ai_report(detected_objects, scene):
    object_list = ", ".join([obj["label"] for obj in detected_objects])

    return f"""
AI Image Analysis Report

Detected Objects:
{object_list}

Predicted Scene:
{scene}

This system automatically analyzes visual content using
computer vision and machine learning techniques and
generates a human-readable report for decision support.
""".strip()
