"""CSV/JSON and publication-table rendering."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ORACLE_COLUMNS = [
    ("m_vIoU", "m-vIoU"), ("vIoU_gt_0", ">0"),
    ("vIoU_gt_0.1", ">0.1"), ("vIoU_gt_0.3", ">0.3"),
    ("vIoU_gt_0.5", ">0.5"),
]
ONE_MIN_COLUMNS = [
    ("m_tIoU", "m-tIoU"), ("tIoU_gt_0", ">0"),
    ("tIoU_gt_0.1", ">0.1"), ("tIoU_gt_0.3", ">0.3"),
    ("tIoU_gt_0.5", ">0.5"),
    ("m_vIoU", "m-vIoU"), ("vIoU_gt_0", ">0"),
    ("vIoU_gt_0.1", ">0.1"), ("vIoU_gt_0.3", ">0.3"),
    ("vIoU_gt_0.5", ">0.5"),
    ("m_vIoU_Int", "m-vIoU-Int"), ("vIoU_Int_gt_0", ">0"),
    ("vIoU_Int_gt_0.1", ">0.1"), ("vIoU_Int_gt_0.3", ">0.3"),
    ("vIoU_Int_gt_0.5", ">0.5"),
]
TG_COLUMNS = ONE_MIN_COLUMNS[:5]
SG_COLUMNS = ONE_MIN_COLUMNS[5:]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _best(scores: list[dict[str, Any]], key: str, tolerance: float = 1e-12) -> set[str]:
    value = max(float(row[key]) for row in scores)
    return {row["model"] for row in scores if abs(float(row[key]) - value) <= tolerance}


def _cell(row: dict[str, Any], key: str, best: set[str]) -> str:
    rendered = f"{100.0 * float(row[key]):.2f}"
    return rf"\textbf{{{rendered}}}" if row["model"] in best else rendered


def latex_table(scores: list[dict[str, Any]], setting: str) -> str:
    oracle = setting == "oracle"
    columns = ORACLE_COLUMNS if oracle else ONE_MIN_COLUMNS
    best = {key: _best(scores, key) for key, _ in columns}
    column_spec = "l" + "r" * len(columns)
    lines = [
        r"\documentclass{article}", r"\usepackage{booktabs}",
        r"\usepackage{graphicx}", r"\usepackage[margin=0.5in]{geometry}",
        r"\pagestyle{empty}", r"\begin{document}", r"\begin{table}[ht]",
        r"\centering", r"\small",
    ]
    if not oracle:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule"])
    if oracle:
        lines.append("Model & " + " & ".join(label for _, label in columns) + r" \\")
    else:
        lines.append(
            r"Model & \multicolumn{5}{c}{Temporal} & "
            r"\multicolumn{5}{c}{Joint STVG} & "
            r"\multicolumn{5}{c}{Spatial within Temporal Intersection} \\"
        )
        lines.append(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}")
        lines.append(" & " + " & ".join(label for _, label in columns) + r" \\")
    lines.append(r"\midrule")
    for row in scores:
        escaped_model = str(row["model"]).replace("_", r"\_")
        values = [_cell(row, key, best[key]) for key, _ in columns]
        lines.append(escaped_model + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if not oracle:
        lines.append("}")
    lines.extend([r"\end{table}", r"\end{document}", ""])
    return "\n".join(lines)


def _fallback_plot(scores: list[dict[str, Any]], setting: str, pdf: Path, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ORACLE_COLUMNS if setting == "oracle" else ONE_MIN_COLUMNS
    headers = ["Model", *[label for _, label in columns]]
    cells = [
        [row["model"], *[f"{100.0 * float(row[key]):.2f}" for key, _ in columns]]
        for row in scores
    ]
    width = 10 if setting == "oracle" else 22
    figure, axis = plt.subplots(figsize=(width, 2.1))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8 if setting == "oracle" else 6.5)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8edf3")
    best = {key: _best(scores, key) for key, _ in columns}
    for row_index, score in enumerate(scores, start=1):
        for column_index, (key, _) in enumerate(columns, start=1):
            if score["model"] in best[key]:
                table[(row_index, column_index)].set_text_props(weight="bold")
    figure.tight_layout()
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_table(scores: list[dict[str, Any]], setting: str, directory: Path) -> dict[str, str]:
    stem = "oracle_scores" if setting == "oracle" else f"{setting}_scores"
    tex, pdf, png = (directory / f"{stem}.{suffix}" for suffix in ("tex", "pdf", "png"))
    tex.write_text(latex_table(scores, setting), encoding="utf-8")
    engine = shutil.which("pdflatex") or shutil.which("xelatex")
    renderer = "matplotlib_fallback"
    if engine:
        completed = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=directory, capture_output=True, text=True,
        )
        if completed.returncode == 0 and pdf.exists() and shutil.which("pdftoppm"):
            prefix = directory / f"{stem}_render"
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(prefix)],
                check=True, capture_output=True,
            )
            prefix.with_suffix(".png").replace(png)
            renderer = Path(engine).name
        else:
            _fallback_plot(scores, setting, pdf, png)
    else:
        _fallback_plot(scores, setting, pdf, png)
    return {"tex": str(tex), "pdf": str(pdf), "png": str(png), "renderer": renderer}


def oracle_role_summary(per_role: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_role:
        grouped[(str(row["model"]), str(row["role"]))].append(float(row["role_vIoU"]))
    result = []
    for (model, role), values in sorted(grouped.items()):
        result.append({
            "model": model,
            "role": role,
            "N": len(values),
            "m_vIoU": sum(values) / len(values),
            **{
                f"vIoU_gt_{threshold:g}": sum(value > threshold for value in values) / len(values)
                for threshold in (0.0, 0.1, 0.3, 0.5)
            },
        })
    return result


def _role_matrix(role_scores: list[dict[str, Any]]):
    models = sorted({str(row["model"]) for row in role_scores})
    roles = sorted({str(row["role"]) for row in role_scores})
    lookup = {(row["role"], row["model"]): row for row in role_scores}
    return models, roles, lookup


def oracle_role_latex(role_scores: list[dict[str, Any]]) -> str:
    models, roles, lookup = _role_matrix(role_scores)
    lines = [
        r"\documentclass{article}", r"\usepackage{booktabs}",
        r"\usepackage{graphicx}", r"\usepackage[margin=0.5in]{geometry}",
        r"\pagestyle{empty}", r"\begin{document}", r"\begin{table}[ht]",
        r"\centering", r"\small", r"\caption{Oracle m-vIoU by semantic role (\%).}",
        rf"\begin{{tabular}}{{lr{'r' * len(models)}}}", r"\toprule",
        "Role & N & " + " & ".join(model.replace("_", r"\_") for model in models) + r" \\",
        r"\midrule",
    ]
    for role in roles:
        values = [float(lookup[(role, model)]["m_vIoU"]) for model in models]
        best = max(values)
        cells = [
            rf"\textbf{{{100 * value:.2f}}}" if abs(value - best) <= 1e-12 else f"{100 * value:.2f}"
            for value in values
        ]
        label = role.replace("_", r"\_")
        lines.append(f"{label} & {lookup[(role, models[0])]['N']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{document}", ""])
    return "\n".join(lines)


def render_oracle_roles(role_scores: list[dict[str, Any]], directory: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = "oracle_role_scores"
    tex, pdf, png = (directory / f"{stem}.{suffix}" for suffix in ("tex", "pdf", "png"))
    tex.write_text(oracle_role_latex(role_scores), encoding="utf-8")
    models, roles, lookup = _role_matrix(role_scores)
    headers = ["Semantic role", "N", *models]
    cells = []
    for role in roles:
        cells.append([
            role,
            str(lookup[(role, models[0])]["N"]),
            *[f"{100 * float(lookup[(role, model)]['m_vIoU']):.2f}" for model in models],
        ])
    figure, axis = plt.subplots(figsize=(12, 8.5))
    axis.axis("off")
    axis.set_title("Oracle m-vIoU by semantic role (%)", fontsize=14, weight="bold", pad=14)
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8edf3")
    for row_index, role in enumerate(roles, start=1):
        values = [float(lookup[(role, model)]["m_vIoU"]) for model in models]
        best = max(values)
        for model_index, value in enumerate(values, start=2):
            if abs(value - best) <= 1e-12:
                table[(row_index, model_index)].set_text_props(weight="bold")
                table[(row_index, model_index)].set_facecolor("#eef6e9")
    figure.tight_layout()
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"tex": str(tex), "pdf": str(pdf), "png": str(png), "renderer": "matplotlib_fallback"}


def _split_latex(scores: list[dict[str, Any]], kind: str) -> str:
    tg = kind == "tg"
    columns = (
        TG_COLUMNS if tg else
        SG_COLUMNS[:5] if kind == "stvg" else
        SG_COLUMNS[5:] if kind == "viou_int" else
        SG_COLUMNS
    )
    best = {key: _best(scores, key) for key, _ in columns}
    title = {
        "tg": "Temporal Grounding",
        "sg": "Spatial Grounding",
        "stvg": "Joint STVG",
        "viou_int": "Spatial Grounding within the Temporal Intersection",
    }[kind]
    lines = [
        r"\documentclass{article}", r"\usepackage{booktabs}",
        r"\usepackage{graphicx}", r"\usepackage[margin=0.5in]{geometry}",
        r"\pagestyle{empty}", r"\begin{document}", r"\begin{table}[ht]",
        r"\centering", r"\small", rf"\caption{{One-minute {title} results (\%).}}",
        rf"\begin{{tabular}}{{l{'r' * len(columns)}}}", r"\toprule",
    ]
    if kind != "sg":
        lines.append("Model & " + " & ".join(label for _, label in columns) + r" \\")
    else:
        lines.append(
            r"Model & \multicolumn{5}{c}{Joint STVG} & "
            r"\multicolumn{5}{c}{Spatial within Temporal Intersection} \\"
        )
        lines.append(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}")
        lines.append(" & " + " & ".join(label for _, label in columns) + r" \\")
    lines.append(r"\midrule")
    for row in scores:
        model = str(row["model"]).replace("_", r"\_")
        lines.append(model + " & " + " & ".join(
            _cell(row, key, best[key]) for key, _ in columns
        ) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{document}", ""])
    return "\n".join(lines)


def _split_plot(
    scores: list[dict[str, Any]], kind: str, pdf: Path, png: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tg = kind == "tg"
    columns = (
        TG_COLUMNS if tg else
        SG_COLUMNS[:5] if kind == "stvg" else
        SG_COLUMNS[5:] if kind == "viou_int" else
        SG_COLUMNS
    )
    title = {
        "tg": "One-minute Temporal Grounding (tIoU, %)",
        "sg": "One-minute Spatial Grounding (vIoU and vIoU-Int, %)",
        "stvg": "One-minute Joint STVG (vIoU, %)",
        "viou_int": "One-minute Spatial Grounding within Temporal Intersection (vIoU-Int, %)",
    }[kind]
    headers = ["Model", *[label for _, label in columns]]
    cells = [
        [row["model"], *[f"{100 * float(row[key]):.2f}" for key, _ in columns]]
        for row in scores
    ]
    figure, axis = plt.subplots(figsize=(16 if kind == "sg" else 10, 2.4))
    axis.axis("off")
    axis.set_title(title, fontsize=13, weight="bold", pad=12)
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5 if kind == "sg" else 8.5)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8edf3")
    best = {key: _best(scores, key) for key, _ in columns}
    for row_index, score in enumerate(scores, start=1):
        for column_index, (key, _) in enumerate(columns, start=1):
            if score["model"] in best[key]:
                table[(row_index, column_index)].set_text_props(weight="bold")
                table[(row_index, column_index)].set_facecolor("#eef6e9")
    figure.tight_layout()
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def render_context_split(
    scores: list[dict[str, Any]], directory: Path, setting: str,
) -> dict[str, dict[str, str]]:
    rendered = {}
    stems = {
        "tg": f"{setting}_tg_scores",
        "sg": f"{setting}_sg_scores",
        "stvg": f"{setting}_stvg_scores",
        "viou_int": f"{setting}_viou_int_scores",
    }
    for kind, stem in stems.items():
        tex, pdf, png = (directory / f"{stem}.{suffix}" for suffix in ("tex", "pdf", "png"))
        tex.write_text(_split_latex(scores, kind), encoding="utf-8")
        _split_plot(scores, kind, pdf, png)
        rendered[kind] = {
            "tex": str(tex), "pdf": str(pdf), "png": str(png),
            "renderer": "matplotlib_fallback",
        }
    return rendered


def render_one_min_split(
    scores: list[dict[str, Any]], directory: Path,
) -> dict[str, dict[str, str]]:
    """Compatibility wrapper; production code uses ``render_context_split``."""
    return render_context_split(scores, directory, "context_1min")
