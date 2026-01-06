"""
Runs the sliding-window co-occurrence graph pipeline separately
for each city category (sea, nosea, north, south).

The script:
- for each existing category corpus, calls run_pipeline_interactive from
  coocurence_window.py with a fixed window size and top-N terms,
- passes a suffix equal to the category name so that all Gephi exports
  are tagged accordingly.

Inputs
------
- data/processed/corpus_sea.json
- data/processed/corpus_nosea.json
- data/processed/corpus_north.json
- data/processed/corpus_south.json
    (cleaned corpora by category, one document per page)

Outputs
-------
For each category where the corpus file exists, the script triggers:
- data/link_analysis/coocurence_window_graph/nodes_{metric}_{category}.csv
- data/link_analysis/coocurence_window_graph/edges_{metric}_{category}.csv
where metric ∈ {"jaccard", "cosine"} and category ∈ {"sea", "nosea", "north", "south"}.
"""
import os
from coocurence_window import run_pipeline_interactive  

CORPUS_FILES = {
    "sea":   "data/processed/corpus_sea.json",
    "nosea": "data/processed/corpus_nosea.json",
    "north": "data/processed/corpus_north.json",
    "south": "data/processed/corpus_south.json",
}

if __name__ == "__main__":
    for group, path in CORPUS_FILES.items():
        if not os.path.exists(path):
            print(f"\n[WARNING] Skipping {group}: file not found -> {path}")
            continue

        print("\n" + "="*70)
        print(f"=== RUNNING PIPELINE FOR GROUP: {group} ===")
        print("="*70)

        run_pipeline_interactive(
            json_path=path,
            window_size=7,
            top_N=10000,
            suffix=group,   
        )
