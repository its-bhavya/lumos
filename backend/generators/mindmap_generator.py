from graphviz import Digraph

def escape_label(text: str) -> str:
    """Escape special characters for Graphviz HTML-like labels."""
    return text.replace("<", "&lt;").replace(">", "&gt;").replace('"', '\\"')

def make_label(title: str, desc: str) -> str:
    """
    Create a safe HTML-like label.
    Only includes description if it's non-empty.
    """
    title = escape_label(title)
    desc = escape_label(desc)
    if desc.strip():
        return f"<<b>{title}</b><br/><font point-size='9'>{desc}</font>>"
    else:
        return f"<<b>{title}</b>>"

def generate_mindmap_svg_from_json(data: dict) -> str:
    """
    Generate a renderable SVG mindmap from JSON.
    Returns SVG string (UTF-8), safe for direct rendering in HTML/React.
    """

    dot = Digraph(comment=f"Mindmap: {data['central_topic']}", format='svg')

    # Graph and node default attributes
    dot.attr(rankdir='LR', bgcolor='white')
    dot.attr(
        'node',
        shape='box',
        style='filled',
        fillcolor='#f0f8ff',
        fontsize='10',
        fontname='Arial'
    )

    # Recursive function to add subtopics
    def add_subtopics(parent_id, subtopics, count):
        for topic in subtopics:
            node_id = f"{parent_id}_{count[0]}"
            count[0] += 1

            label = make_label(topic["title"], topic.get("description", ""))
            dot.node(node_id, label=label)
            dot.edge(parent_id, node_id)

            if topic.get("children"):
                add_subtopics(node_id, topic["children"], count)

    # Central node
    central_id = "central"
    dot.node(
        central_id,
        f"<<b>{escape_label(data['central_topic'])}</b>>",
        shape='ellipse',
        fillcolor='#ffebcd',
        fontsize='13',
        fontname='Arial'
    )

    # Add all subtopics
    add_subtopics(central_id, data.get("subtopics", []), count=[1])

    # Generate SVG in-memory
    svg_bytes = dot.pipe(format='svg')
    svg_text = svg_bytes.decode("utf-8")

    # Optional: compact SVG for embedding
    svg_clean = svg_text.replace('\r', '').replace('\n', '')

    return svg_clean
