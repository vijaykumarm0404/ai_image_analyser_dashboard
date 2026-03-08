import matplotlib.pyplot as plt
from collections import Counter

def generate_object_chart(detected_objects):
    if not detected_objects:
        return None

    labels = [obj["label"] for obj in detected_objects]
    counts = Counter(labels)

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set_title("Detected Object Frequency (Current Image)")
    ax.set_xlabel("Object")
    ax.set_ylabel("Count")

    return fig