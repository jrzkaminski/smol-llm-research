import requests
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import hamming
from itertools import combinations
import warnings
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

warnings.filterwarnings("ignore")


def str2dict(s):
    """
    Dictionary parsing from ToolAlpaca
    """
    s = s[s.find('{"') + 2 : s.rfind('"}')]
    s = s.split('.", ')
    d = dict()
    for param in s:
        key, value = param.split('": "')
        key = key.replace('"', "")
        value = value.replace('"', "")
        d[key] = value
    return d


def get_funcs(funcs_desc):
    """
    Extract functions from ToolAlpaca, including their descriptions and parameters.
    """
    funcs_preprocessed = dict()
    for func_name, description in funcs_desc.items():
        params = dict()
        output = ""
        sequences = description.split("\n")
        for seq in sequences:
            if "Parameters: " in seq and "{}" not in seq:
                params = str2dict(seq.replace("Parameters: ", ""))
            if "Output: " in seq:
                output = seq.replace("Output: ", "")
        if len(params) > 0 or len(output) > 0:
            funcs_preprocessed[func_name] = {"Parameters": params, "Output": output}
    return funcs_preprocessed


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embeddings(params, output):
    """
    Extract embedding from parameter and output descriptions.
    """
    sentences = params + [output]
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings[:-1], sentence_embeddings[-1]


def encode_funcs(funcs_preprocessed):
    """
    Applies get_embeddings function to dictionary from get_funcs function.
    """
    all_params_encoded, all_outputs_encoded = [], []
    for func, data in funcs_preprocessed.items():
        params = []
        for param, desc in data["Parameters"].items():
            text = ". ".join([param, desc])
            params.append(text)
        output = data["Output"]
        params_encoded, output_encoded = get_embeddings(params, output)
        all_params_encoded.append(params_encoded)
        all_outputs_encoded.append(output_encoded)
    return all_params_encoded, all_outputs_encoded


def get_cosine_similarity(emb1, emb2):
    cos_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return cos_sim


def get_bonds(all_params_encoded, all_outputs_encoded, threshold):
    """
    Calculates cosine proximity between all extracted embeddings.
    Generates a list of edges depending on threshold.
    This list contains only function and parameter IDs.
    """
    bonds = []  # [[input_func_id, output_func_id, input_param_id, cos_sim], ...]
    similarities = []
    for input_func_id in range(len(all_outputs_encoded)):
        for param_id in range(len(all_params_encoded[input_func_id])):
            for output_func_id in range(len(all_outputs_encoded)):
                if output_func_id != input_func_id:
                    cosine_similarity = get_cosine_similarity(
                        all_params_encoded[input_func_id][param_id],
                        all_outputs_encoded[output_func_id],
                    )
                    similarities.append(cosine_similarity)
                    bonds.append(
                        [input_func_id, output_func_id, param_id, cosine_similarity]
                    )
    if similarities:
        similarities = np.array(similarities).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_similarities = scaler.fit_transform(similarities).flatten()
        for i in range(len(bonds)):
            bonds[i][3] = scaled_similarities[i]

    final_bonds = []
    for i in range(len(bonds)):
        if bonds[i][3] > threshold:
            final_bonds.append(bonds[i])

    return final_bonds


def decode_bonds(bonds, funcs_preprocessed):
    """
    Decodes function and parameter IDs from get_bonds dunction into their names.
    """
    func_names, params_names = [], []
    for func, data in funcs_preprocessed.items():
        params = [param for param in data["Parameters"].keys()]
        params_names.append(params)
        func_names.append(func)
    bonds_encoded = []
    for bond in bonds:
        bonds_encoded.append(
            [
                func_names[bond[0]],
                func_names[bond[1]],
                params_names[bond[0]][bond[2]],
                bond[3],
            ]
        )
    return bonds_encoded


def gradient_color(t):
    light_blue = (173 / 255, 216 / 255, 230 / 255)  # RGB for light blue
    dark_blue = (0, 0, 139 / 255)  # RGB for dark blue

    r = light_blue[0] * (1 - t) + dark_blue[0] * t
    g = light_blue[1] * (1 - t) + dark_blue[1] * t
    b = light_blue[2] * (1 - t) + dark_blue[2] * t

    return (r, g, b)


def draw_graph(graph, ax, title, pos):
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    edge_colors = [gradient_color(weight**3) for weight in edge_weights]
    edge_widths = [2 + (weight**3) for weight in edge_weights]

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


def adjacency_matrix_similarity(G1, G2):
    """
    Сравнение графов на основе их матриц смежности.
    Возвращает долю совпадающих элементов.
    """
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()

    if A1.shape != A2.shape:
        return 0

    return np.sum(A1 == A2) / A1.size


def degree_distribution_similarity(G1, G2):
    """
    Сравнение графов на основе распределения степеней вершин.
    Используется косинусное сходство.
    """
    deg1 = sorted([d for n, d in G1.out_degree()])
    deg2 = sorted([d for n, d in G2.out_degree()])

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
    A1 = nx.adjacency_matrix(G1).todense().flatten()
    A2 = nx.adjacency_matrix(G2).todense().flatten()

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
    L1 = nx.laplacian_matrix(G1).todense()
    L2 = nx.laplacian_matrix(G2).todense()

    if L1.shape != L2.shape:
        return 0

    eigvals1 = np.linalg.eigvals(L1)
    eigvals2 = np.linalg.eigvals(L2)

    return np.linalg.norm(np.sort(eigvals1) - np.sort(eigvals2))


def graph_edit_distance_similarity(G1, G2):
    """
    Сравнение графов с использованием расстояния редактирования графа.
    """
    try:
        ged = nx.graph_edit_distance(G1, G2)
        return 1 / (1 + ged)
    except nx.NetworkXError:
        return 0


def create_directed_graph(edges, nodes):
    """
    Построение направленного графа с фиксированными весами, равными 1
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    for edge in edges:
        source, target = edge
        graph.add_edge(source, target, weight=1)

    return graph


def create_weighted_directed_graph(edges, nodes):
    """
    Построение направленного взвешенного графа
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    for edge in edges:
        source, target, _, weight = edge
        graph.add_edge(source, target, weight=weight)

    return graph


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


with open("ToolAlpaca_Modeling/ToolAlpaca.json", "r") as file:
    data = json.load(file)

with open("ToolAlpaca_Modeling/graphs_16-60.json", "r") as file:
    edges16_60 = json.load(file)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

edges = edges16_60
# 0..15 + 16
# 15..30 + 46 - 15

num_nodes = []
best_thresholds = []
for i in range(30):
    bias = 16 if i < 15 else 31
    metrics_list = []
    G2_list = []
    G1 = create_directed_graph(edges[i]["edges"], edges[i]["nodes"])
    thresholds = np.arange(0, 1.05, 0.05)
    for threshold in thresholds:
        api_id = i + bias
        funcs = get_funcs(data[api_id]["Function_Description"])
        params, outputs = encode_funcs(funcs)
        bonds = get_bonds(params, outputs, threshold)
        decoded_bonds = decode_bonds(bonds, funcs)
        G2 = create_weighted_directed_graph(decoded_bonds, list(funcs.keys()))

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
    print("API name:", edges[i]["api_name"])
    print("number of nodes:", len(edges[i]["nodes"]))
    print("best threshold:", thresholds[best_idx])
    print("metrics:")
    print(json.dumps(metrics_list[best_idx], indent=4))
    print("----------------------------")
    num_nodes.append(len(edges[i]["nodes"]))
    best_thresholds.append(thresholds[best_idx])
    try:
        position = nx.spring_layout(G1, k=1.5)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(edges[i]["api_name"])
        draw_graph(G1, axes[0], title="Reference graph", pos=position)
        draw_graph(G2_list[best_idx], axes[1], title="Constructed graph", pos=position)
        plt.savefig(f"ToolAlpaca_Modeling/images/{edges[i]['api_name']}.png")
        plt.show()
    except Exception as e:
        print(e)

print(num_nodes)
print(best_thresholds)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=num_nodes, y=best_thresholds, color="b")
plt.xlabel("number of nodes")
plt.ylabel("best threshold")
plt.show()
