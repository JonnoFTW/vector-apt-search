#!/usr/bin/env python3
import argparse

import numpy as np
from datetime import datetime
from pathlib import Path
from platformdirs import user_cache_dir
from sentence_transformers import SentenceTransformer

# needs libapt-pkg-dev
import apt

parser = argparse.ArgumentParser(description="Apt Search with Human String")
parser.add_argument("query", metavar="QUERY", type=str, nargs=1, help="Query string")
parser.add_argument("-k", "--top-k", type=int, help="Number of results", default=5)
args = parser.parse_args()

if args.top_k <= 0:
    raise ValueError("-k --top-k must be a positive integer")
apt_cache_path = Path("/var/cache/apt/pkgcache.bin")
if not apt_cache_path.exists():
    print("No apt cache found")
    exit(1)

apt_cache_last_updated = datetime.fromtimestamp(apt_cache_path.stat().st_mtime)

embedding_cache = Path(f"{user_cache_dir()}/apt-vector/apt-vectors-cache.npz")
embedding_cache.parent.mkdir(parents=True, exist_ok=True)
vector_cache_last_updated = datetime.fromtimestamp(
    embedding_cache.stat().st_mtime if embedding_cache.exists() else 0
)

descriptions = []
cache = apt.Cache(memonly=True)
for pkg in iter(cache):
    descriptions.append([pkg.name, pkg.versions[0].description])
descriptions = np.array(descriptions)


if not embedding_cache.exists() or apt_cache_last_updated > vector_cache_last_updated:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Creating apt description vectors")
    embeddings = model.encode(
        descriptions[:, 1],
        batch_size=256,
        show_progress_bar=True,
    )
    print("Writing cache to", embedding_cache)
    np.savez(embedding_cache, embeddings)
else:
    # use the cached vector embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", )
    print("Loading cached apt description vectors")
    embeddings = np.load(embedding_cache)["arr_0"]
embedded_query = model.encode(args.query, show_progress_bar=False)

similarities = model.similarity_pairwise(embedded_query, embeddings).numpy()
indexes = np.argsort(similarities)
top_k_indexes = indexes[-args.top_k :][::-1]
top_k_packages = descriptions[:, 0][indexes]


for idx in top_k_indexes:
    pkg_name = descriptions[idx][0]
    score = similarities[idx]
    pkg = cache.get(pkg_name)
    if pkg:
        version = pkg.versions[0]
        print("Name:      ", pkg.name)
        print("Score:     ", round(score, 2))
        print("Version:   ", version.version)
        print("Installed: ", pkg.is_installed)
        print("Summary:   ", version.raw_description)
        print("\n")
