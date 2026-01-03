#!/usr/bin/env python3
import os
from typing import Tuple

from similarity import cosine_similarity

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import pickle
import hashlib

# needs libapt-pkg-dev python3-apt cudnn9-cuda
import apt
import humanize
import numpy as np
from light_embed import TextEmbedding
from platformdirs import user_cache_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-apt-search")


@dataclass
class PackageDescriptionHash:
    description: str
    hash_val: bytes


def get_apt_cache() -> Tuple[apt.Cache, dict[str, PackageDescriptionHash]]:
    """
    Reads the apt cache and returns a dictionary of package names to their descriptions and hashes.
    :return:
    """
    descriptions = {}
    cache = apt.Cache(memonly=True)
    for pkg in iter(cache):
        description = pkg.versions[0].description
        descriptions[pkg.name] = PackageDescriptionHash(
            description, hashlib.sha256(description.encode()).digest()
        )
    return cache, descriptions


def update_vector_cache(descriptions: dict[str, PackageDescriptionHash], refresh: bool = False):
    """
    Update the package description vectors in the cache.
    :param descriptions:
    :param refresh:
    :return:
    """
    apt_cache_path = Path("/var/cache/apt/pkgcache.bin")
    if not apt_cache_path.exists():
        print("No apt cache found")
        exit(1)

    embedding_cache_dir = Path(f"{user_cache_dir()}/apt-vector")
    description_cache_file = Path(embedding_cache_dir / "gpu-apt-description-cache.pkl")
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    # check against the pickled description cache
    if description_cache_file.exists():
        with description_cache_file.open("rb") as f:
            descriptions_cached = pickle.load(f)
    else:
        descriptions_cached = {}

    # check if the cached description hash matches that in the apt cache, if it differs, then add it to be recalculated
    names_to_calculate = []
    descriptions_to_calculate = []
    for name, pkg_hash in descriptions.items():
        if name not in descriptions_cached or pkg_hash.hash_val != descriptions_cached[name][1]:
            names_to_calculate.append(name)
            descriptions_to_calculate.append(pkg_hash.description)

    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    if refresh or descriptions_to_calculate:
        logger.debug(
            "Creating apt description vectors for %d packages", len(descriptions_to_calculate)
        )
        start = datetime.now()
        embeddings = model.encode(
            descriptions_to_calculate,
            batch_size=256,
        )
        logger.debug("Finished calculating vectors in %s", datetime.now() - start)
        logger.debug("Writing cache to %s", description_cache_file)
        for name, embedding in zip(names_to_calculate, embeddings):
            descriptions_cached[name] = (
                descriptions[name].description,
                descriptions[name].hash_val,
                embedding,
            )
        with description_cache_file.open("wb") as f:
            pickle.dump(descriptions_cached, f)
    return descriptions_cached, model


def search_by_vector(model: TextEmbedding, query: str, descriptions_cached, top_k: int):
    """
    Returns the top k packages that are most similar to the query string.
    :param model:
    :param query:
    :param descriptions_cached:
    :param top_k:
    :return:
    """
    names = np.array([n for n in descriptions_cached.keys()])
    embeddings = np.array([e for _, _, e in descriptions_cached.values()])
    logger.debug("Embeddings size: %s", humanize.naturalsize(embeddings.nbytes))
    embedded_query = model.encode(query, return_as_array=True)

    similarities = cosine_similarity(embedded_query[0], embeddings)
    indexes = np.argsort(similarities)

    top_k_indexes = indexes[-top_k:][::-1]
    return zip(names[top_k_indexes], similarities[top_k_indexes])


def show_packages(packages, cache: apt.Cache, verbose: bool):
    for pkg_name, score in packages:
        pkg = cache.get(pkg_name)
        score = round(float(score), 2)
        if pkg:
            version = pkg.versions[0]
            print(
                f"\033[92m{pkg.name}\033[0m/{version.origins[0].codename} score={score} {version.architecture}{' [installed]' if pkg.is_installed else ''}"
            )
            print(f" ", version.description if verbose else version.raw_description)
            print()


def main():
    parser = argparse.ArgumentParser(description="Apt Search with Human String")
    parser.add_argument("query", metavar="QUERY", type=str, nargs=1, help="Query string")
    parser.add_argument("-k", "--top-k", type=int, help="Number of results", default=5)
    parser.add_argument(
        "-r",
        "--refresh",
        action="store_true",
        help="Force refresh apt cache description vectors",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.top_k <= 0:
        raise ValueError("-k --top-k must be a positive integer")

    apt_cache, descriptions = get_apt_cache()
    descriptions_cached, model = update_vector_cache(descriptions, refresh=args.refresh)

    packages = search_by_vector(model, args.query, descriptions_cached, args.top_k)
    show_packages(packages, apt_cache, args.verbose)


if __name__ == "__main__":
    main()
