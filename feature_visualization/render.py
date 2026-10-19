"""Portable offline HTML and conventional publication exports."""
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .analysis import fit_patch_pca, scale_rgb
from .common import write_json
from .dense import legend_html, mask_view, render_patch_page


def palette(labels):
    names = sorted(set(v for v in labels if v is not None))
    cmap = colormaps["tab10" if len(names) <= 10 else "tab20"]
    colors = {name: to_hex(cmap(i % cmap.N)) for i, name in enumerate(names)}
    colors[None] = "#9ca3af"
    return [colors[v] for v in labels], colors


def prepare_assets(root, records):
    root = Path(root)
    thumbnails = []
    samples = []
    for record in records:
        thumbnails.append("data:image/jpeg;base64," + base64.b64encode(Path(record["thumbnail"]).read_bytes()).decode())
        samples.append({key: record.get(key) for key in ("sample_id", "image", "label_name", "partition", "width", "height")})
    (root / "assets.js").write_text("window.VIZ_SAMPLES=" + json.dumps(samples).replace("<", "\\u003c") + ";\nwindow.VIZ_THUMBS=" + json.dumps(thumbnails) + ";\n")
    from plotly.offline import get_plotlyjs
    (root / "plotly.min.js").write_text(get_plotlyjs())


def render_scatter(directory, root, title, coordinates, records, clusters, method, metadata):
    import plotly.graph_objects as go
    directory, root = Path(directory), Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    dim = coordinates.shape[1]
    label_values = [r.get("label_name") for r in records]
    cluster_values = [f"Cluster {int(v)}" for v in clusters]
    label_colors, labels_legend = palette(label_values)
    cluster_colors, cluster_legend = palette(cluster_values)
    stem = f"{method}_{dim}d"
    custom = np.arange(len(records))
    descriptions = [f"{html.escape(Path(r['image']).name)}<br>{html.escape(r.get('label_name') or 'Unlabeled')}<br>{cluster_values[i]}" for i, r in enumerate(records)]
    arguments = dict(x=coordinates[:, 0], y=coordinates[:, 1], customdata=custom, text=descriptions,
                     mode="markers", marker=dict(color=label_colors, size=5 if dim == 2 else 3, opacity=.8), hovertemplate="%{text}<extra></extra>")
    trace = go.Scattergl(**arguments) if dim == 2 else go.Scatter3d(z=coordinates[:, 2], **arguments)
    fig = go.Figure(trace)
    axis = {"title": f"{method.upper()} component 1"}
    fig.update_layout(title=f"{title} | {method.upper()} {dim}D", template="plotly_white", margin=dict(l=45, r=10, t=60, b=45),
                      xaxis=axis, yaxis={"title": f"{method.upper()} component 2"},
                      scene=dict(xaxis_title=f"{method.upper()} 1", yaxis_title=f"{method.upper()} 2", zaxis_title=f"{method.upper()} 3"))
    div_id = "feature-plot"
    chart = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, default_height="78vh", config={"responsive": True})
    toggles = dict(labels=label_colors, clusters=cluster_colors)
    legends = {"labels": {str(k) if k is not None else "Unlabeled": v for k, v in labels_legend.items() if k is not None or None in label_values},
               "clusters": {str(k): v for k, v in cluster_legend.items() if k is not None}}
    # All point positions remain fixed when switching colors. Sample text uses textContent.
    script = """
const colorSets=COLORS, legends=LEGENDS;
const graph=document.getElementById('feature-plot');
function changeColor(mode){
 Plotly.restyle(graph,{'marker.color':[colorSets[mode]]});
 document.getElementById('legend').replaceChildren();
 for(const [name,color] of Object.entries(legends[mode])){
  const row=document.createElement('div');const dot=document.createElement('span');
  dot.className='dot';dot.style.background=color;row.append(dot,document.createTextNode(name));
  document.getElementById('legend').append(row);
 }
 document.querySelectorAll('button[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
}
function showSample(point){
 const i=Number(point.customdata), s=window.VIZ_SAMPLES[i];if(!s)return;
 document.getElementById('thumb').src=window.VIZ_THUMBS[i];
 document.getElementById('thumb').hidden=false;
 document.getElementById('sample-label').textContent=s.label_name||'Unlabeled';
 document.getElementById('sample-path').textContent=s.image;
 document.getElementById('sample-id').textContent=s.sample_id;
 document.getElementById('sample-split').textContent='Split: '+s.partition+' | '+s.width+' x '+s.height;
}
graph.on('plotly_click',event=>showSample(event.points[0]));
document.querySelectorAll('button[data-mode]').forEach(b=>b.addEventListener('click',()=>changeColor(b.dataset.mode)));
changeColor('labels');
""".replace("COLORS", json.dumps(toggles)).replace("LEGENDS", json.dumps(legends))
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TITLE</title><script src="../plotly.min.js"></script><script src="../assets.js"></script>
<style>body{font:15px system-ui;margin:0;background:#f5f7fa;color:#172334}header{padding:18px 26px;background:#102b42;color:white}a{color:inherit}main{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px;padding:18px}.chart{background:white;border-radius:12px;min-width:0}#feature-plot{height:78vh}aside{background:white;border-radius:12px;padding:18px;overflow-wrap:anywhere}button{border:1px solid #a6b4c2;background:white;padding:9px 12px;border-radius:7px;cursor:pointer;margin:4px 3px 12px 0}.active{background:#dbeefa;border-color:#1678ab}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px}#legend div{padding:3px 0}#thumb{width:100%;max-height:220px;object-fit:contain;background:#edf2f6}small{color:#65758a}p{line-height:1.5}@media(max-width:850px){main{grid-template-columns:1fr}#feature-plot{height:65vh}}</style></head>
<body><header><a href="../index.html">SeaDino feature explorer</a> / TITLE</header><main><section class="chart">CHART</section>
<aside><strong>Color points by</strong><br><button data-mode="labels">Dataset labels</button><button data-mode="clusters">K-means clusters</button><div id="legend"></div><hr><h3>Selected image</h3><img id="thumb" alt="Selected sample thumbnail" hidden><p id="sample-label">Click a point to inspect its image.</p><small id="sample-split"></small><p id="sample-path"></p><small id="sample-id"></small><hr><p><small>One point represents one full image. Positions use backbone features; labels only set colors. Each model has its own projection. Distances between separate model plots are not comparable.</small></p></aside></main><script>SCRIPT</script></body></html>"""
    page = page.replace("TITLE", html.escape(f"{title} — {method.upper()} {dim}D")).replace("CHART", chart).replace("SCRIPT", script)
    (directory / f"{stem}.html").write_text(page)
    with (directory / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "image", "label", "cluster"] + [f"component_{i+1}" for i in range(dim)])
        for i, row in enumerate(records):
            writer.writerow([row["sample_id"], row["image"], row.get("label_name"), int(clusters[i]), *coordinates[i].tolist()])
    write_json(directory / f"{stem}.json", metadata)
    for coloring, names, colors in (("labels", label_values, label_colors), ("clusters", cluster_values, cluster_colors)):
        static = plt.figure(figsize=(9, 7))
        ax = static.add_subplot(111, projection="3d" if dim == 3 else None)
        for name in sorted(set(v or "Unlabeled" for v in names)):
            mask = np.array([(v or "Unlabeled") == name for v in names])
            points = coordinates[mask]
            arguments = dict(s=12, alpha=.75, c=np.array(colors)[mask].tolist(), label=name, rasterized=True)
            if dim == 3:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], **arguments)
            else:
                ax.scatter(points[:, 0], points[:, 1], **arguments)
        ax.set_title(f"{title} | {method.upper()} {dim}D | {coloring}")
        ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
        if dim == 3:
            ax.set_zlabel("Component 3")
        ax.legend(loc="best", fontsize=8)
        static.tight_layout()
        for extension in ("png", "pdf"):
            static.savefig(directory / f"{stem}_{coloring}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(static)
    return dict(method=method, dimensions=dim, page=f"{directory.name}/{stem}.html")


def render_patches(directory, records, patches, geometries, components, seed, title):
    directory = Path(directory) / "patches"
    directory.mkdir(parents=True, exist_ok=True)
    if not patches:
        return []
    projected, metadata = fit_patch_pca(patches, components, seed)
    np.savez_compressed(directory / "pca_basis.npz", mean=metadata.pop("mean"), basis=metadata.pop("basis"))
    write_json(directory / "pca.json", metadata)
    results = []
    for scores, geometry in zip(projected, geometries):
        row = records[geometry["index"]]
        gh, gw = geometry["grid_h"], geometry["grid_w"]
        tensor = torch.from_numpy(scores.reshape(gh, gw, 3)).permute(2, 0, 1).unsqueeze(0)
        # Interpolate float components before robust scaling; crop off padded border afterward.
        dense = F.interpolate(tensor, size=(geometry["padded_height"], geometry["padded_width"]), mode="bicubic", align_corners=False)
        dense = dense[0, :, :geometry["height"], :geometry["width"]].permute(1, 2, 0).numpy()
        rgb = scale_rgb(dense, metadata["color_limits"])
        heat = colormaps["viridis"](scale_rgb(dense, metadata["color_limits"])[..., 0])[..., :3]
        with Image.open(row["staged_image"]) as raw:
            original = np.asarray(raw.convert("RGB").resize((geometry["width"], geometry["height"]), Image.Resampling.BICUBIC)) / 255.
        overlay = .55 * original + .45 * rgb
        stem = row["sample_id"][:20]
        rgb_path = directory / f"{stem}_rgb.png"
        for suffix, image in (("original", original), ("heatmap", heat), ("rgb", rgb), ("overlay", overlay)):
            Image.fromarray(np.round(image * 255).clip(0, 255).astype(np.uint8)).save(directory / f"{stem}_{suffix}.png")
        panels = [("Original", original), (f"PC {components[0]}", heat), (f"PCA RGB {components}", rgb), ("Feature overlay", overlay)]
        if row.get("mask_path"):
            mask, edges = mask_view(row, (geometry["width"], geometry["height"]))
            outlined = rgb.copy()
            outlined[edges] = 1.
            panels = [("Original", original), ("Ground-truth semantic mask", mask),
                      (f"PC {components[0]}", heat), (f"PCA RGB {components}", rgb),
                      ("PCA + ground-truth boundaries", outlined), ("Feature overlay", overlay)]
            Image.fromarray((outlined * 255).round().clip(0, 255).astype(np.uint8)).save(directory / f"{stem}_boundaries.png")
        columns = 3 if row.get("mask_path") else 4
        rows = (len(panels) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.1 * rows), squeeze=False)
        for ax, (label, image) in zip(axes.flat, panels):
            ax.imshow(image); ax.set_title(label); ax.axis("off")
        fig.suptitle(f"{title} | {row.get('label_name') or row['partition']} | {Path(row['image']).name}")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{stem}_comparison.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        results.append(dict(sample_id=row["sample_id"], index=geometry["index"], rgb=str(rgb_path)))
    render_patch_page(directory.parent, records, results, title)
    return results


def render_comparisons(root, records, results):
    root = Path(root)
    directory = root / "patch_comparisons"
    directory.mkdir(exist_ok=True)
    by_sample = {}
    for result in results:
        for patch in result.get("patch_maps", []):
            by_sample.setdefault(patch["index"], []).append((result["name"], patch["rgb"]))
    pages = []
    for index, comparisons in by_sample.items():
        row = records[index]
        panels = [("Original", row["staged_image"])]
        if row.get("mask_color"):
            panels.append(("Ground truth (official palette)", row["mask_color"]))
        panels += comparisons
        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (name, filename) in zip(axes.flat, panels):
            with Image.open(filename) as image:
                ax.imshow(image)
            ax.set_title(name)
        fig.suptitle(f"{row.get('label_name') or row['partition']} | {Path(row['image']).name}\nPCA colors are independent across models")
        fig.tight_layout()
        stem = row["sample_id"][:20]
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{stem}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        pages.append(f"patch_comparisons/{stem}.png")
    return pages


def render_index(root, run, results, comparisons):
    root = Path(root)
    from .quality_guide import write_quality_guide
    write_quality_guide(root, run)
    attention_path = root / "attention_run.json"
    attention_state = json.loads(attention_path.read_text()) if attention_path.exists() else {}
    attention = attention_state.get("results", {})
    attention_by_name = {item["name"]: item for item in attention.values() if item.get("state") == "complete"}
    blocks = []
    for result in results:
        metrics = result["metrics"]
        links = " ".join(f'<a href="{html.escape(p["page"])}">{p["method"].upper()} {p["dimensions"]}D</a>' for p in result["plots"])
        key = next((k for k, value in run["results"].items() if value["name"] == result["name"]), None)
        if key and (root / key / "patches.html").exists():
            links += f' <a href="{html.escape(key)}/patches.html"><strong>Spatial feature maps</strong></a>'
        if result["name"] in attention_by_name:
            links += f' <a href="{html.escape(attention_by_name[result["name"]]["page"])}"><strong>Attention maps</strong></a>'
        def metric(name):
            value = metrics.get(name)
            return "—" if value is None else f"{value:.3f}"
        details = "Frozen backbone · semantic masks are references" if run["source"]["task"] == "segmentation" else f'Cosine silhouette: {metric("silhouette_cosine")} · ARI: {metric("adjusted_rand_index")} · NMI: {metric("normalized_mutual_information")}'
        blocks.append(f'<article><h2>{html.escape(result["name"])}</h2><p>{links or "Patch maps only"}</p><small>{details}</small></article>')
    completed_names = {result["name"] for result in results}
    for key, spec in run["models"].items():
        if spec.get("name", key) not in completed_names:
            blocks.append(f'<article><h2>{html.escape(spec.get("name", key))}</h2><p>Feature maps queued</p><small>This model has not finished extraction yet.</small></article>')
    images = ""
    for n, p in enumerate(comparisons):
        row = next((r for r in run["records"] if Path(p).stem == r["sample_id"][:20]), {})
        with Image.open(root / p) as image:
            width, height = image.size
        images += f'<a href="{p}"><img width="{width}" height="{height}" style="width:100%;height:auto" loading="{"eager" if n == 0 else "lazy"}" src="{p}" alt="Backbone patch PCA comparison"></a>'
        if row.get("mask_path"):
            images += legend_html(row)
    resolution = run.get("arguments", {}).get("image_size", 224)
    longest = run.get("arguments", {}).get("patch_longest", 512)
    own_images = f'''<details style="background:white;padding:18px;border-radius:12px;margin:18px 0"><summary><strong>Use your own images / choose resolution</strong></summary><p>Current resolution: <strong>{resolution} × {resolution}</strong> for whole-image features; <strong>{longest}px longest side</strong> for patch PCA. Padding to each model's patch multiple is recorded in model metadata.</p><p>This HTML is a viewer. Run a new extraction in your SeaDino WSL terminal:</p><pre style="white-space:pre-wrap">cd /home/njan320/SeaDino
conda activate SeaDino
python -m feature_visualization --input_dir /mnt/d/Neel/my_images --models all --image_size 448 --patch_longest 768 --batch_size 2</pre><p>For one image, replace <code>--input_dir /mnt/d/Neel/my_images</code> with <code>--image_path /mnt/d/Neel/my_images/fish.jpg --patch_images 1</code>. A single image gives patch maps and a one-point scatter; use multiple images to inspect grouping.</p><p>Windows <code>D:\\Neel\\my_images</code> is <code>/mnt/d/Neel/my_images</code> in WSL. Subfolders such as <code>fish/</code> and <code>coral/</code> provide optional labels. New reports are saved under <code>D:\\Neel\\FeatureVisualizations</code>.</p></details>'''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SeaDino backbone feature explorer</title><style>body{{font:16px system-ui;background:#f4f7fa;color:#18283c;margin:0}}header{{padding:28px 5%;background:#102b42;color:white}}main{{max-width:1250px;margin:auto;padding:24px}}.models{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}article{{background:white;border-radius:12px;padding:20px}}h2{{font-size:20px}}a{{color:#0874a7;margin-right:12px;line-height:2}}small{{color:#52677a}}.patches img{{width:100%;margin:15px 0;border-radius:10px}}p{{line-height:1.6}}</style></head><body><header><h1>SeaDino backbone feature explorer</h1><p>{html.escape(run['source_name'])} · {len(run['records'])} shared images · seed {run['seed']} · frozen backbones</p></header><main><p>One point represents one whole image. Labels supply colors; backbone features determine positions. Select a point to view its thumbnail. Each model has an independent projection. Plots provide qualitative evidence alongside probe/kNN results.</p><p>Sampling: {html.escape(json.dumps(run['sample_summary']['labels']))}. Split: {html.escape(run['source'].get('split', 'folder'))}. Cluster metrics use the original normalized features. Class count sets default K; class labels do not fit projections or K-means.</p><section class="models">{''.join(blocks)}</section><h2>Patch PCA comparisons</h2><p>Each model uses a shared PCA basis across these images. Colors represent feature directions, not semantic class predictions, and do not match across models.</p><section class="patches">{images}</section><p><a href="run.json">Run provenance</a><a href="metrics.csv">Metrics CSV</a><a href="samples.csv">Sample metadata</a></p></main></body></html>'''
    page = page.replace('<section class="models">', own_images + '<section class="models">')
    guide = '<p><a href="quality_guide.html"><strong>How to assess backbone quality</strong></a> · Clustering metrics below are not classification accuracy.</p>'
    guide += f'<p><strong>Feature maps: {len(results)}/{len(run["models"])} models ready.</strong></p>'
    if not attention_state and run.get("arguments", {}).get("attention_maps"):
        guide += '<p><strong>Attention queued:</strong> attention extraction starts after the spatial feature pass.</p>'
    if (root / "attention_comparisons/index.html").exists():
        guide += '<p><a href="attention_comparisons/index.html"><strong>Compare attention across all models</strong></a> — CLS and received-patch attention are labeled separately.</p>'
    if attention_state:
        guide += f'<p><strong>Attention: {len(attention_by_name)}/{len(run["models"])} models ready.</strong> State: {html.escape(attention_state.get("state", "unknown"))}. Open the Attention maps link inside each completed model card.</p>'
    page = page.replace('<section class="models">', guide + '<section class="models">')
    page = page.replace('shared images · seed', f'shared images · global {resolution}×{resolution} / patch {longest}px · seed')
    if run["source"]["task"] == "segmentation":
        start = page.index('<main><p>')
        end = page.index('<details', start)
        intro = f'<main><p><strong>Dense spatial feature comparison.</strong> The same {len(run["records"])} {html.escape(run["source"].get("split", ""))} images are shown for every model. Patch PCA uses an aspect-preserved {longest}px longest side. Masks use nearest-neighbor resizing to the same unpadded geometry.</p><p>Ground truth uses semantic class colors and a shared legend; PCA colors are feature directions fitted without labels. Inspect boundaries and regions visually. These panels are not segmentation predictions or mIoU scores.</p>'
        page = page[:start] + intro + page[end:]
        page = page.replace(f'global {resolution}×{resolution} / patch {longest}px', f'spatial patch maps {longest}px')
        page = page.replace('Clustering metrics below are not classification accuracy.', 'Use dense prediction scores for quantitative segmentation quality.')
        page = page.replace('<a href="metrics.csv">Metrics CSV</a>', '')
        if (root / "semantic_labels.json").exists():
            page = page.replace('<a href="run.json">', '<a href="semantic_labels.json">Semantic classes / palette</a><a href="run.json">')
    related = [item for item in run.get("related_reports", []) if (root / item["path"]).is_file()]
    if related:
        navigation = '<nav style="padding:16px;background:white;border-radius:12px;margin:18px 0"><strong>Other dataset:</strong> ' + ''.join(f'<a href="{html.escape(item["path"])}">{html.escape(item["name"])}</a>' for item in related) + '</nav>'
        page = page.replace('<details', navigation + '<details', 1)
    if run.get("corrections"):
        notes = ''.join(f'<p><strong>{html.escape(item["state"].capitalize())}:</strong> {html.escape(item["description"])} {html.escape(item.get("benchmark_note", ""))}</p>' for item in run["corrections"])
        page = page.replace('<details', '<aside style="padding:16px;background:#fff8de;border-radius:12px;margin:18px 0">'+notes+'</aside><details',1)
    if (root/"attention_diagnostics/index.html").exists():
        page = page.replace('<section class="models">','<p><a href="attention_diagnostics/index.html"><strong>Weight-loading and raw-grid audit</strong></a></p><section class="models">',1)
    (root / "index.html").write_text(page)
    keys = ("model", "silhouette_cosine", "adjusted_rand_index", "normalized_mutual_information", "clusters_effective")
    with (root / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for result in results:
            writer.writerow({"model": result["name"], **{k: result["metrics"].get(k) for k in keys[1:]}})
