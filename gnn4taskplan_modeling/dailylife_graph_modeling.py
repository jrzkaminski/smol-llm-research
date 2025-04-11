import json
import torch
import torch.nn.functional as F
import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import hamming
import warnings

warnings.filterwarnings("ignore")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_cosine_similarity(emb1, emb2):
    cos_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return cos_sim


def parse_funcs_from_api(api_data):
    funcs_preprocessed = {}
    for node in api_data["nodes"]:
        func_name = node["id"]
        func_desc = node["desc"]
        params_dict = {}
        for p in node["parameters"]:
            param_name = p["name"]
            param_desc = p["desc"]
            params_dict[param_name] = param_desc

        funcs_preprocessed[func_name] = {"Parameters": params_dict, "Output": func_desc}
        # print(funcs_preprocessed[func_name])

    return funcs_preprocessed


def get_embeddings(sentences, tokenizer, model):
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def encode_funcs(funcs_preprocessed, tokenizer, model):
    all_params_encoded = []
    all_outputs_encoded = []
    for func, data in funcs_preprocessed.items():
        param_texts = []
        for param_name, param_desc in data["Parameters"].items():
            text = f"{param_name}. {param_desc}"
            param_texts.append(text)

        output_text = data["Output"]

        combined_texts = param_texts + [output_text]
        embeddings = get_embeddings(combined_texts, tokenizer, model)

        params_embeddings = embeddings[:-1]
        output_embedding = embeddings[-1]

        all_params_encoded.append(params_embeddings)
        all_outputs_encoded.append(output_embedding)

    return all_params_encoded, all_outputs_encoded


def get_bonds(all_params_encoded, all_outputs_encoded, threshold):
    bonds = []
    similarities = []
    for input_func_id in range(len(all_outputs_encoded)):
        for param_id in range(len(all_params_encoded[input_func_id])):
            param_emb = all_params_encoded[input_func_id][param_id]
            for output_func_id in range(len(all_outputs_encoded)):
                if output_func_id != input_func_id:
                    output_emb = all_outputs_encoded[output_func_id]
                    cos_sim = get_cosine_similarity(param_emb, output_emb)
                    similarities.append(cos_sim)
                    bonds.append([input_func_id, output_func_id, param_id, cos_sim])

    if similarities:
        # similarities = np.array(similarities).reshape(-1, 1)
        # scaler = MinMaxScaler()
        # scaled_similarities = scaler.fit_transform(similarities).flatten()
        # for i in range(len(bonds)):
        #     bonds[i][3] = scaled_similarities[i]

        for i in range(len(bonds)):
            bonds[i][3] = similarities[i]

    final_bonds = []
    for b in bonds:
        if b[3] > threshold:
            final_bonds.append(b)
    return final_bonds


def decode_bonds(bonds, funcs_preprocessed):
    func_names = list(funcs_preprocessed.keys())
    params_names = []
    for func, data in funcs_preprocessed.items():
        p_list = list(data["Parameters"].keys())
        params_names.append(p_list)

    bonds_encoded = []
    for bond in bonds:
        in_func_id, out_func_id, param_id, weight = bond
        in_func_name = func_names[in_func_id]
        out_func_name = func_names[out_func_id]
        in_param_name = params_names[in_func_id][param_id]
        bonds_encoded.append([in_func_name, out_func_name, in_param_name, weight])
    return bonds_encoded


def create_directed_graph(ref_graph_data):
    G = nx.DiGraph()
    for node in ref_graph_data["nodes"]:
        G.add_node(node["id"])

    for link in ref_graph_data["links"]:
        src = link["source"]
        tgt = link["target"]
        G.add_edge(src, tgt, weight=1)

    return G


def create_weighted_directed_graph(decoded_bonds, funcs_preprocessed):
    G = nx.DiGraph()
    func_names = list(funcs_preprocessed.keys())

    for fn in func_names:
        G.add_node(fn)

    for b in decoded_bonds:
        source, target, param_name, weight = b
        G.add_edge(source, target, weight=weight)

    return G


def adjacency_matrix_similarity(G1, G2):
    """
    Сравнение графов на основе их матриц смежности.
    Возвращает долю совпадающих элементов.
    """
    A1 = nx.adjacency_matrix(G1, nodelist=sorted(G1.nodes())).todense()
    A2 = nx.adjacency_matrix(G2, nodelist=sorted(G2.nodes())).todense()

    if A1.shape != A2.shape:
        return 0
    return np.sum(A1 == A2) / A1.size


def degree_distribution_similarity(G1, G2):
    """
    Сравнение графов на основе распределения степеней вершин.
    Используется косинусное сходство.
    """
    deg1 = sorted([d for _, d in G1.out_degree()])
    deg2 = sorted([d for _, d in G2.out_degree()])

    if len(deg1) != len(deg2):
        return 0

    dot_product = np.dot(deg1, deg2)
    norm1 = np.linalg.norm(deg1)
    norm2 = np.linalg.norm(deg2)

    return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0


def hamming_distance_similarity(G1, G2):
    """
    Сравнение графов на основе расстояния Хэмминга между векторами ребер.
    """

    nodes_G1 = sorted(G1.nodes())
    nodes_G2 = sorted(G2.nodes())
    A1 = nx.adjacency_matrix(G1, nodelist=nodes_G1).todense().flatten()
    A2 = nx.adjacency_matrix(G2, nodelist=nodes_G2).todense().flatten()

    if A1.shape != A2.shape:
        return 0
    return 1 - hamming(A1, A2)


def jaccard_similarity(G1, G2):
    """
    Сравнение графов на основе коэффициента Жаккара для множеств рёбер.
    """
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())

    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)
    return intersection / union if union > 0 else 0


def spectral_similarity(G1, G2):
    """
    Сравнение графов на основе их спектральных характеристик (собственных значений).
    """
    G1_und = G1.to_undirected()
    G2_und = G2.to_undirected()

    L1 = nx.laplacian_matrix(G1_und, nodelist=sorted(G1_und.nodes())).todense()
    L2 = nx.laplacian_matrix(G2_und, nodelist=sorted(G2_und.nodes())).todense()

    if L1.shape != L2.shape:
        return float("inf")
    eigvals1 = np.linalg.eigvals(L1)
    eigvals2 = np.linalg.eigvals(L2)
    return np.linalg.norm(np.sort(eigvals1) - np.sort(eigvals2))


def graph_edit_distance_similarity(G1, G2):
    """
    Сравнение графов с использованием расстояния редактирования графа.
    """
    try:
        ged = nx.graph_edit_distance(G1, G2)
        if ged is None:
            return 0
        return 1 / (1 + ged)
    except nx.NetworkXError:
        return 0


def get_optimal_idx(metrics_list):
    """
    Выбор индекса оптимального threshold
    """
    metrics_matrix = np.array(
        [
            [
                m.get("Adjacency Matrix", 0),
                m.get("Degree Distribution", 0),
                m.get("Hamming Distance", 0),
                m.get("Jaccard Similarity", 0),
                m.get("Graph Edit Distance", 0),
                1 / (1 + m.get("Spectral Similarity", 0)),
            ]
            for m in metrics_list
        ],
        dtype=float,
    )

    scaler = MinMaxScaler()
    normalized_matrix = scaler.fit_transform(metrics_matrix)
    sums = normalized_matrix.sum(axis=1)
    max_val = np.max(sums)
    idx = np.where(sums == max_val)[0][-1]
    return idx


def gradient_color(t):
    light_blue = (173 / 255, 216 / 255, 230 / 255)
    dark_blue = (0, 0, 139 / 255)
    r = light_blue[0] * (1 - t) + dark_blue[0] * t
    g = light_blue[1] * (1 - t) + dark_blue[1] * t
    b = light_blue[2] * (1 - t) + dark_blue[2] * t
    return (r, g, b)


def draw_graph(graph, ax, title, pos):
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    edge_colors = [gradient_color(w**3) for w in edge_weights]
    edge_widths = [2 + (w**3) for w in edge_weights]

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=800,
        node_color="lightblue",
        font_size=10,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="white",
        connectionstyle="arc3,rad=0.0",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=4,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        connectionstyle="arc3,rad=0.2",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        ax=ax,
    )


if __name__ == "__main__":
    api_name = "dailylife"
    with open(
        f"gnn4taskplan_modeling/{api_name}/tool_desc.json", "r", encoding="utf-8"
    ) as file:
        tool_data = json.load(file)
    with open(
        f"gnn4taskplan_modeling/{api_name}/graph_desc.json", "r", encoding="utf-8"
    ) as file:
        ref_graph_data = json.load(file)

    funcs_preprocessed = parse_funcs_from_api(tool_data)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    all_params_encoded, all_outputs_encoded = encode_funcs(
        funcs_preprocessed, tokenizer, model
    )

    G1 = create_directed_graph(ref_graph_data)

    thresholds = np.arange(0.5, 1.05, 0.05)
    metrics_list = []
    G2_list = []

    for th in thresholds:
        bonds = get_bonds(all_params_encoded, all_outputs_encoded, th)
        decoded_bonds = decode_bonds(bonds, funcs_preprocessed)
        G2 = create_weighted_directed_graph(decoded_bonds, funcs_preprocessed)

        metrics = {
            "Adjacency Matrix": adjacency_matrix_similarity(G1, G2),
            "Degree Distribution": degree_distribution_similarity(G1, G2),
            "Hamming Distance": hamming_distance_similarity(G1, G2),
            "Jaccard Similarity": jaccard_similarity(G1, G2),
            "Spectral Similarity": spectral_similarity(G1, G2),
            "Graph Edit Distance": graph_edit_distance_similarity(G1, G2),
        }
        metrics_list.append(metrics)
        G2_list.append(G2)

    best_idx = get_optimal_idx(metrics_list)
    best_threshold = thresholds[best_idx]
    print("best threshold: ", best_threshold)
    print("metrics:")
    print(json.dumps(metrics_list[best_idx], indent=4))

    try:
        position = nx.spring_layout(G1, k=1.5)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(api_name)
        draw_graph(G1, axes[0], "Reference Graph", pos=position)
        draw_graph(G2_list[best_idx], axes[1], "Constructed Graph", pos=position)
        plt.savefig(f"gnn4taskplan_modeling/dailylife/constructed.png")
        plt.show()
    except Exception as e:
        print(e)

    final_bonds = get_bonds(all_params_encoded, all_outputs_encoded, best_threshold)
    decoded_bonds_final = decode_bonds(final_bonds, funcs_preprocessed)

    links_list = []
    for b in decoded_bonds_final:
        source, target, param_name, weight = b
        item = {
            "source": source,
            "target": target,
            "type": "complete",
        }
        if item not in links_list:
            links_list.append(item)

    result_graph_structure = {"nodes": tool_data["nodes"], "links": links_list}

    # print(json.dumps(result_graph_structure, indent=2, ensure_ascii=False))
    # print()
    with open(
        f"gnn4taskplan_modeling/{api_name}/constructed_graph.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result_graph_structure, f, indent=2, ensure_ascii=False)
