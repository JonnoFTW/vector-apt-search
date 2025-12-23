#!/usr/bin/env python3
import argparse
import logging
import humanize
import numpy as np
from datetime import datetime
from pathlib import Path
from platformdirs import user_cache_dir
from light_embed import TextEmbedding

# needs libapt-pkg-dev python3-apt
import apt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-apt-search")

parser = argparse.ArgumentParser(description="Apt Search with Human String")
parser.add_argument("query", metavar="QUERY", type=str, nargs=1, help="Query string")
parser.add_argument("-k", "--top-k", type=int, help="Number of results", default=5)
parser.add_argument("-r", "--refresh", action='store_true', help="Force refresh apt cache description vectors")
parser.add_argument("-v", "--verbose", action='store_true', help="Verbose output")
args = parser.parse_args()

if args.top_k <= 0:
    raise ValueError("-k --top-k must be a positive integer")
apt_cache_path = Path("/var/cache/apt/pkgcache.bin")
if not apt_cache_path.exists():
    print("No apt cache found")
    exit(1)

if args.verbose:
    logger.setLevel(logging.DEBUG)

apt_cache_last_updated = datetime.fromtimestamp(apt_cache_path.stat().st_mtime)

embedding_cache = Path(f"{user_cache_dir()}/apt-vector/apt-vectors-cache.npz")
embedding_cache.parent.mkdir(parents=True, exist_ok=True)
vector_cache_last_updated = datetime.fromtimestamp(
    embedding_cache.stat().st_mtime if embedding_cache.exists() else 0
)

names = []
descriptions = []
cache = apt.Cache(memonly=True)
for pkg in iter(cache):
    names.append(pkg.name)
    descriptions.append(pkg.versions[0].description)


model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", )
if args.refresh or (not embedding_cache.exists() or apt_cache_last_updated > vector_cache_last_updated):
    logger.debug("Creating apt description vectors")
    start = datetime.now()
    embeddings = model.encode(
        descriptions,
        batch_size=256,
    )
    logger.debug("Finished calculating vectors in %s", datetime.now() - start)
    logger.debug("Writing cache to %s", embedding_cache)
    np.savez(embedding_cache, embeddings)
else:
    # use the cached vector embeddings
    logger.debug("Loading cached apt description vectors\n")
    embeddings = np.load(embedding_cache)["arr_0"]

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two NumPy vectors.
    """
    dot_product = np.dot(vec2, vec1)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2, axis=1)

    return dot_product / (norm_vec1 * norm_vec2)
logger.debug("Embeddings size:", humanize.naturalsize(embeddings.nbytes))
embedded_query = model.encode(args.query, return_as_array=True)

similarities = cosine_similarity(embedded_query[0], embeddings)
indexes = np.argsort(similarities)
top_k_indexes = indexes[-args.top_k :][::-1]

for idx in top_k_indexes:
    pkg_name = names[idx]
    score = similarities[idx]
    pkg = cache.get(pkg_name)
    if pkg:
        version = pkg.versions[0]
        print("Name:      ", pkg.name)
        print("Score:     ", round(score, 2))
        print("Version:   ", version.version)
        print("Installed: ", pkg.is_installed)
        print("Summary:   ", version.description if args.verbose else version.raw_description)
        print()