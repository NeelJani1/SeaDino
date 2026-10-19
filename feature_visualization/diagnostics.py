"""CPU-only inspection of genuine attention grids and display interpolation."""
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image

from .attention import dense_map
from .common import write_json


def color(values, low, high):
    scaled = np.clip((values-low)/max(high-low,1e-20),0,1)
    return (colormaps["magma"](scaled)[...,:3]*255).round().astype(np.uint8)


def write_diagnostics(root):
    root = Path(root)
    run = json.loads((root/"run.json").read_text())
    attention = json.loads((root/"attention_run.json").read_text())
    if attention["state"] != "complete" or any(v.get("state") != "complete" for v in attention["results"].values()):
        raise ValueError("Finish all attention refreshes before producing the loading/rendering audit")
    directory = root/"attention_diagnostics"
    directory.mkdir(exist_ok=True)
    weight_audit_path = root.parent/"fathomnet_all9_20261019/attention_loading_diagnostics.json"
    audit = json.loads(weight_audit_path.read_text())
    cards, data, measurements = [],[],{}
    for key,spec in run["models"].items():
        item = attention["results"][key]["items"][0]
        folder = root/key/"attention"
        metadata = json.loads((folder/item["metadata"]).read_text())
        geometry = metadata["geometry"]
        with np.load(folder/item["arrays"]) as arrays:
            weights = arrays["last"].copy()
        grid = weights.reshape(geometry["grid_h"],geometry["grid_w"])
        nearest = np.array(Image.fromarray(grid).resize((geometry["padded_width"],geometry["padded_height"]),Image.Resampling.NEAREST))[:geometry["height"],:geometry["width"]]
        interpolated = dense_map(weights,geometry)
        limits = metadata["display_limits"]["last"]
        images = {}
        for mode,values in (("nearest",nearest),("bilinear",interpolated)):
            for scale,(low,high) in {"percentile":(limits["low"],limits["high"]),"zero":(0,float(weights.max()))}.items():
                filename = f"{key}_{mode}_{scale}.png"
                Image.fromarray(color(values,low,high)).save(directory/filename)
                images[mode+"_"+scale] = filename
        original = os.path.relpath(item["original"],directory)
        probability = weights/weights.sum()
        entropy = float(-np.sum(probability*np.log(np.maximum(probability,1e-30)))/np.log(len(probability)))
        measurements[key] = dict(first_sample_id=item["sample_id"],grid=[geometry["grid_h"],geometry["grid_w"]],
            query=metadata["query"],normalized_spatial_entropy=entropy,raw_spatial_mass=float(weights.sum()))
        data.append(dict(key=key,original=original,images=images))
        cards.append(f'<article><h2>{html.escape(spec["name"])}</h2><p>{geometry["grid_w"]} × {geometry["grid_h"]} patches · spatial entropy {entropy:.3f}</p><canvas id="canvas-{key}" width="{geometry["width"]}" height="{geometry["height"]}"></canvas><p>{html.escape(metadata["query"])}</p><a href="../{key}/attention/{item["heads"]}">Individual heads</a></article>')
    table = ""
    for key in ("seadino_s","seadino_b"):
        weights = audit["weights"][key]
        corr = audit["attention_pairs"][key]
        table += f'<tr><td>{html.escape(run["models"][key]["name"])}</td><td>{weights["existing_loaded_tensor_audit"]["matched_tensors"]}/211 exactly match teacher</td><td>{weights["tensors_different_from_foundation"]}/211 differ from base</td><td>{100*weights["relative_weight_l2"]:.2f}%</td><td>{corr["pearson_mean"]:.3f}</td></tr>'
    paired_panels = ''
    for base,adapted in (("dinov3_s","seadino_s"),("dinov3_b","seadino_b")):
        maps = []
        for key in (base,adapted):
            item = attention["results"][key]["items"][0]
            folder = root/key/"attention"
            details = json.loads((folder/item["metadata"]).read_text())
            with np.load(folder/item["arrays"]) as arrays:
                maps.append(dense_map(arrays["last"],details["geometry"]))
        high = max(float(maps[0].max()),float(maps[1].max()))
        difference = maps[1]-maps[0]
        fig,axes = plt.subplots(1,3,figsize=(14,4))
        axes[0].imshow(maps[0],cmap='magma',vmin=0,vmax=high);axes[0].set_title(run['models'][base]['name'])
        axes[1].imshow(maps[1],cmap='magma',vmin=0,vmax=high);axes[1].set_title(run['models'][adapted]['name'])
        bound = max(float(np.abs(difference).max()),1e-12)
        plot = axes[2].imshow(difference,cmap='seismic',vmin=-bound,vmax=bound);axes[2].set_title('Difference: SeaDino − DINOv3')
        fig.colorbar(plot,ax=axes[2],shrink=.6,label='Attention probability difference')
        for ax in axes:ax.axis('off')
        fig.suptitle('Shared zero-to-peak scale within each pair; saved normalization policies')
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(directory/f'{adapted}_difference.{extension}',dpi=160,bbox_inches='tight')
        plt.close(fig)
        paired_panels += f'<img style="width:100%;height:auto" src="{adapted}_difference.png" alt="Base versus adapted attention with shared scale and signed difference">'
    correction = ""
    archive = root/"corrections/sam_rectangular_rope_before"
    if (archive/"attention_run.json").exists():
        old = json.loads((archive/"attention_run.json").read_text())["results"]["sam31"]["items"][0]
        new = attention["results"]["sam31"]["items"][0]
        geometry = json.loads((root/"sam31/attention"/new["metadata"]).read_text())["geometry"]
        with np.load(archive/"attention"/old["arrays"]) as arrays:
            before = arrays["last"].copy()
        with np.load(root/"sam31/attention"/new["arrays"]) as arrays:
            after = arrays["last"].copy()
        high = max(float(before.max()),float(after.max()))
        original = np.array(Image.open(new["original"]).convert("RGB"))/255.
        panels = [("Original",original)]
        for label,values in (("SAM before coordinate correction",before),("SAM corrected row-major coordinates",after)):
            heat = color(dense_map(values,geometry),0,high)/255.
            panels.append((label,.65*original+.35*heat))
        fig,axes = plt.subplots(1,3,figsize=(15,4))
        for ax,(label,value) in zip(axes,panels):
            ax.imshow(value);ax.set_title(label,fontsize=10);ax.axis("off")
        fig.suptitle("Same weights / image / resolution; shared zero-to-peak attention scale")
        fig.tight_layout()
        for extension in ("png","pdf"):
            fig.savefig(directory/f"sam_before_after.{extension}",dpi=160,bbox_inches="tight")
        plt.close(fig)
        correction = '<h2>SAM rectangular-coordinate correction</h2><p>The previous adapter used grid height as the flattened row stride. It now uses x=index%width and y=index//width, preserving the square-input behavior and existing isotropic scale. The former parity test shared the same adapter and could not detect the coordinate error. Independent meshgrid tests now cover landscape, portrait and square inputs. Earlier SAM segmentation scores with rectangular inputs require fresh evaluation; square classification features are unchanged.</p><img style="width:100%;height:auto" src="sam_before_after.png" alt="SAM attention before and after coordinate correction">'
        measurements["sam31"]["before_after_attention_pearson"] = float(np.corrcoef(before,after)[0,1])
    script = '''const entries=DATA, loaded=new Map();
async function load(path){if(!loaded.has(path)){loaded.set(path,new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=path;}));}return loaded.get(path);}
async function draw(){const mode=document.querySelector('#mode').value,scale=document.querySelector('#scale').value,alpha=Number(document.querySelector('#opacity').value)/100;document.querySelector('#opacity-label').textContent=Math.round(alpha*100)+'%';await Promise.all(entries.map(async item=>{const canvas=document.querySelector('#canvas-'+item.key),ctx=canvas.getContext('2d'),[original,heat]=await Promise.all([load(item.original),load(item.images[mode+'_'+scale])]);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.globalAlpha=1;ctx.imageSmoothingEnabled=false;ctx.drawImage(original,0,0);ctx.globalAlpha=alpha;ctx.drawImage(heat,0,0);ctx.globalAlpha=1;}));}
document.querySelectorAll('select,input').forEach(x=>x.addEventListener('input',()=>draw().catch(console.error)));draw().catch(console.error);'''.replace("DATA",json.dumps(data).replace("<","\\u003c"))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attention loading and rendering audit</title><style>body{{font:16px system-ui;line-height:1.5;color:#18283c;background:#f4f7fa;max-width:1400px;margin:25px auto;padding:0 20px}}a{{color:#0874a7}}table{{border-collapse:collapse;background:white}}td,th{{border:1px solid #ccd5df;padding:10px;text-align:left}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px}}article,.controls{{background:white;padding:18px;border-radius:10px;margin:16px 0}}canvas{{width:100%;height:auto}}select,input{{margin-right:18px}}.controls{{position:sticky;top:0;z-index:1;border:1px solid #ccd5df}}h2{{font-size:21px}}</style></head><body><a href="../index.html">Back to report</a><h1>Attention loading and rendering audit</h1><p>These controls change only the display of saved attention. Zero overlay opacity shows the original image. Nearest-neighbor mode reveals the actual patch grid; bilinear mode reveals how interpolation makes smooth blobs. Zero-to-peak scaling avoids clipping the brightest 2% as the original percentile display does.</p><h2>SeaDino weights are distinct and loaded exactly</h2><table><tr><th>Model</th><th>Loaded teacher weights</th><th>Compared with foundation</th><th>Relative weight L2 change</th><th>Mean raw attention correlation</th></tr>{table}</table><p>Attention correlations above use 20 FathomNet training images under the saved normalization policies (B uses BenthicNet normalization). No paired maps are identical. Weight distance and attention correlation are descriptive diagnostics, not performance scores.</p><p>DINOv2/CLIP can show irregular attention. The selected DINOv2 checkpoints have no register tokens. The <a href="https://arxiv.org/abs/2309.16588">register paper</a> and <a href="https://arxiv.org/abs/2506.08010">test-time-register paper</a> discuss outlier-token artifacts. CLIP here has strong patch-norm outliers; that does not establish the sole cause of each bright patch. The saved final DINOv2 features do not show comparably large norm outliers.</p><p>MarineInst averages received attention over every spatial query and all heads. This view is diffuse (mean normalized spatial entropy 0.977 on the 20 FathomNet images), and interpolation softens it further. It differs from the learned-CLS query of DINO/SeaDino/CLIP. SAM uses received attention too. These views are not prompted segmentation masks or direct model-quality rankings.</p>{correction}<h2>Inspect the same image across all nine models</h2><div class="controls"><label>Interpolation <select id="mode"><option value="nearest">Actual patch blocks</option><option value="bilinear">Bilinear smoothing</option></select></label><label>Contrast <select id="scale"><option value="zero">Zero to peak</option><option value="percentile">2nd–98th percentiles</option></select></label><label>Overlay <input id="opacity" type="range" min="0" max="100" value="35"><span id="opacity-label">35%</span></label></div><section class="cards">{''.join(cards)}</section><p><a href="measurements.json">Current raw-grid measurements</a> · <a href="{html.escape(os.path.relpath(weight_audit_path,directory))}">Full loading/overlay audit</a></p><script>{script}</script></body></html>'''
    page = page.replace('<h2>Inspect the same image across all nine models</h2>', '<h2>Base versus adapted maps with a shared display scale</h2>'+paired_panels+'<h2>Inspect the same image across all nine models</h2>')
    pre_norm_path = weight_audit_path.parent/"dinov2_pre_norm_diagnostics.json"
    if pre_norm_path.exists():
        checks = json.loads(pre_norm_path.read_text())
        extra = '<p>A fresh forward on the representative FathomNet frame also checked tokens before the final output LayerNorm: '
        extra += '; '.join(f'{html.escape(run["models"][key]["name"])} maximum norm / median = {value["maximum_norm_over_median"]:.2f}×, with cached output error {value["maximum_cached_feature_error"]:.0f}' for key,value in checks.items())
        extra += '. This example does not establish a large-norm-outlier explanation for DINOv2. Irregular attention alone is not evidence of wrong weights.</p>'
        page = page.replace('<h2>SAM rectangular-coordinate correction</h2>',extra+'<h2>SAM rectangular-coordinate correction</h2>')
    (directory/"index.html").write_text(page)
    write_json(directory/"measurements.json",measurements)
    print(f"DIAGNOSTICS: {directory/'index.html'}",flush=True)


if __name__ == "__main__":
    import sys
    write_diagnostics(sys.argv[1])
