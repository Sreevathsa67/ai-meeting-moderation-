from flask import Flask, request, render_template_string
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import json
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# --- Load Model (this will run once when the app starts) ---
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.")

# --- HTML Template for the entire page ---
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>AI Meeting Moderator</title>
    <style>
        body { font-family: sans-serif; margin: 2em; background-color: #f4f4f9; }
        h1, h2 { color: #333; }
        textarea { width: 100%; height: 200px; margin-bottom: 1em; }
        input[type=submit] { background-color: #4CAF50; color: white; padding: 14px 20px; border: none; cursor: pointer; }
        .results { background-color: white; padding: 1em; border-radius: 8px; margin-top: 2em; }
        .topic { border-left: 5px solid #4CAF50; padding-left: 1em; margin-bottom: 1em; }
        .off-topic { border-left: 5px solid #f44336; padding-left: 1em; margin-bottom: 1em; }
    </style>
</head>
<body>
    <h1>AI Meeting Moderator</h1>
    <form method="post">
        <h2>Paste Chat JSON Here:</h2>
        <textarea name="chat_json">{{ sample_json }}</textarea>
        <input type="submit" value="Analyze Conversation">
    </form>

    {% if results %}
    <div class="results">
        <h2>Analysis Results</h2>
        {% for cluster in results.clusters %}
            {% if cluster.label != -1 %}
            <div class="topic">
                <h3>Topic {{ cluster.label }} ({{ cluster.count }} messages)</h3>
                <p><strong>Keywords:</strong> {{ cluster.keywords|join(', ') }}</p>
            </div>
            {% endif %}
        {% endfor %}

        {% for cluster in results.clusters %}
            {% if cluster.label == -1 %}
            <div class="off-topic">
                <h3>Off-Topic / Noise ({{ cluster.count }} messages)</h3>
                <p>These messages did not fit into a main topic.</p>
                 <ul>
                    {% for msg in cluster.examples %}
                    <li>{{ msg.user }}: "{{ msg.text|truncate(80) }}"</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        {% endfor %}

        <h2>Conversation Map</h2>
        <div id="plot">{{ results.plot_html|safe }}</div>
    </div>
    {% endif %}
</body>
</html>
"""

def analyze_conversation(messages):
    # 1. Get Embeddings
    texts = [m.get("text", "") for m in messages]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # 2. Cluster Embeddings
    # Use safe parameters that work for small numbers of messages
    dbscan = DBSCAN(eps=0.7, min_samples=2, metric='cosine')
    labels = dbscan.fit_predict(embeddings)

    # 3. Reduce Dimensions for Plotting
    # Use safe n_neighbors that is less than the number of samples
    n_neighbors = max(2, min(15, len(texts) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    # 4. Extract Topics and Keywords
    cluster_info = {}
    for idx, label in enumerate(labels):
        if label not in cluster_info:
            cluster_info[label] = {'indices': [], 'texts': []}
        cluster_info[label]['indices'].append(idx)
        cluster_info[label]['texts'].append(texts[idx])

    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    output_clusters = []
    for label, data in cluster_info.items():
        keywords = []
        if label != -1 and len(data['texts']) > 1:
            X = vec.fit_transform(data['texts'])
            mean_scores = X.mean(axis=0).A1
            terms = np.array(vec.get_feature_names_out())
            top_idx = np.argsort(mean_scores)[::-1][:5]
            keywords = terms[top_idx].tolist()
        
        examples = [{"user": messages[i].get("user"), "text": texts[i]} for i in data['indices'][:3]]
        output_clusters.append({
            "label": int(label),
            "count": len(data['indices']),
            "keywords": keywords,
            "examples": examples
        })

    # 5. Create Plotly Visualization
    fig = go.Figure()
    for label, data in cluster_info.items():
        points = emb_2d[data['indices']]
        hover_texts = [f"<b>{messages[i]['user']}:</b><br>{messages[i]['text']}" for i in data['indices']]
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers',
            hoverinfo='text',
            text=hover_texts,
            marker=dict(size=10, opacity=0.8),
            name=f'Topic {label}' if label != -1 else 'Off-topic'
        ))
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return {"clusters": output_clusters, "plot_html": plot_html}


@app.route('/', methods=['GET', 'POST'])
def home():
    # Load sample JSON to pre-populate the text box
    with open('sample_chat.json', 'r') as f:
        sample_json = f.read()

    if request.method == 'POST':
        chat_json_str = request.form.get('chat_json')
        try:
            data = json.loads(chat_json_str)
            messages = data.get("messages", [])
            if not messages:
                return "Error: No messages found in JSON.", 400
            
            analysis_results = analyze_conversation(messages)
            return render_template_string(HTML_TEMPLATE, sample_json=chat_json_str, results=analysis_results)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format.", 400
    
    return render_template_string(HTML_TEMPLATE, sample_json=sample_json, results=None)

if __name__ == '__main__':
    app.run(debug=True)