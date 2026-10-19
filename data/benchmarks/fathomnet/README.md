# FathomNet Benchmark (Exploration Status: Postponed)

## Status Note
FathomNet was investigated as a potential downstream testing benchmark. However, as noted in the community and by the original dataset authors:
* The official FathomNet does not provide a direct monolithic image archive or direct downloadable image crops on Hugging Face.
* Access requires querying the FathomNet API over HTTP for individual image framegrabs hosted at MBARI (`database.fathomnet.org`) and performing instance bounding-box / SAM segmentation cropping.
* Because dynamic per-sample HTTP fetching over tens of thousands of frames introduces network bottlenecks, rate limiting, and reproducibility challenges, **FathomNet evaluation is postponed**.

## Adopted Alternative
The SeaDino evaluation suite instead adopts **[DeepFish](file:///home/njan320/SeaDino/data/benchmarks/deepfish/README.md)** (*Saleh et al., Nature Scientific Reports 2020*):
* Monolithic 7.1 GB dataset archive hosted directly on Hugging Face (`Alzayats/DeepFish`).
* 39,766 high-resolution underwater frames across 20 distinct marine habitats.
* Official train/val/test splits for fish classification, counting, and segmentation.
* Directly evaluated via `python run_eval.py --datasets deepfish`.
