import os
from coocurence_window import run_pipeline_interactive  

# creating coocurence graph  and community per categories

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
