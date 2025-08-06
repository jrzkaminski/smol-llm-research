#!/usr/bin/env python3
# combine_results_and_score.py
"""
Объединяет predicted_edges из трёх файлов результатов LLM
и пересчитывает Jaccard similarity с ground-truth графом.
"""

import json
from pathlib import Path
from typing import Set, FrozenSet

# ────────────────────────────────
# Файлы с результатами моделей
# ────────────────────────────────
RESULT_FILES = [
    Path("/home/ilya/smol-llm-research/graph_gen/llm_mixed_results_25batch.json"),
    Path("/home/ilya/smol-llm-research/graph_gen/llm_mixed_results_50batch.json"),
    Path("/home/ilya/smol-llm-research/graph_gen/llm_hierarchical_checkpoint.json"),
]

# ────────────────────────────────
# Файлы с описаниями инструментов
# ────────────────────────────────
DATA_DIR = Path("data/toollinkos")  # core_tools.json, regular_tools.json

# ────────────────────────────────
# Итоговый файл
# ────────────────────────────────
OUTPUT_FILE = Path(
    "/home/ilya/smol-llm-research/graph_gen/llm_mixed_results_combined.json"
)


# ────────────────────────────────
# Вспомогательные функции
# ────────────────────────────────
def jaccard_similarity(a: Set[FrozenSet[str]], b: Set[FrozenSet[str]]) -> float:
    """J(A,B)=|A∩B| / |A∪B| — 0.0 … 1.0."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def load_predicted_edges(result_path: Path) -> Set[FrozenSet[str]]:
    """
    Читает поле predicted_edges и возвращает
    set(frozenset({src, tgt})) без повторов и учёта порядка.
    Работает как с тройками [src, tgt, why], так и с парами [src, tgt].
    """
    data = json.loads(result_path.read_text())
    return {frozenset(edge[:2]) for edge in data.get("predicted_edges", [])}


def load_ground_truth() -> Set[FrozenSet[str]]:
    """Собирает ground-truth зависимости из core_tools + regular_tools."""
    core = json.loads((DATA_DIR / "core_tools.json").read_text())
    regular = json.loads((DATA_DIR / "regular_tools.json").read_text())
    all_tools = core + regular

    gt: Set[FrozenSet[str]] = {
        frozenset((tool["name"], dep["name"]))
        for tool in all_tools
        for dep in tool.get("depends_on", [])
    }
    return gt


# ────────────────────────────────
# Главная логика
# ────────────────────────────────
def main() -> None:
    # 1. Объединяем все предсказанные рёбра из трёх файлов
    combined_edges: Set[FrozenSet[str]] = set()
    for path in RESULT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл: {path}")
        combined_edges |= load_predicted_edges(path)

    # 2. Загружаем ground-truth
    ground_truth = load_ground_truth()

    # 3. Считаем Jaccard similarity
    jac = jaccard_similarity(combined_edges, ground_truth)

    # 4. Печатаем краткий отчёт
    print("── Итоговая статистика ──")
    print(f"Предсказанных рёбер (объединено) : {len(combined_edges)}")
    print(f"Ground-truth рёбер               : {len(ground_truth)}")
    print(f"Jaccard similarity               : {jac:.4f}")

    # 5. Сохраняем результат
    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "combined_predicted_edges_count": len(combined_edges),
                "ground_truth_edges_count": len(ground_truth),
                "jaccard_similarity": jac,
                "combined_predicted_edges": [list(edge) for edge in combined_edges],
            },
            indent=2,
        )
    )
    print(f"Результат сохранён → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
