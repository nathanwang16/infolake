#!/usr/bin/env python3
"""
Random sampler for creating test crawl datasets from marginalia dumps.
Samples N random entries from a source tar archive and creates a new tar.
"""

import argparse
import logging
import random
import sys
import tarfile
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def sample_tar(source_tar: Path, output_tar: Path, sample_size: int, seed: int | None = None) -> None:
    """
    Randomly sample entries from source tar and create a new tar with samples.
    
    Args:
        source_tar: Path to source tar archive
        output_tar: Path for output sampled tar archive
        sample_size: Number of samples to extract
        seed: Random seed for reproducibility (optional)
    
    Raises:
        FileNotFoundError: If source tar does not exist
        ValueError: If sample_size exceeds available entries
    """
    if not source_tar.exists():
        raise FileNotFoundError(f"Source tar not found: {source_tar}")
    
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    
    logger.info(f"Opening source tar: {source_tar}")
    
    with tarfile.open(source_tar, "r:*") as src_tar:
        # Get all member names (excluding directories)
        all_members = [m for m in src_tar.getmembers() if m.isfile()]
        total_count = len(all_members)
        
        logger.info(f"Total files in source: {total_count}")
        
        if sample_size > total_count:
            raise ValueError(
                f"Sample size ({sample_size}) exceeds available entries ({total_count})"
            )
        
        # Random sample
        sampled_members = random.sample(all_members, sample_size)
        logger.info(f"Sampled {sample_size} entries")
        
        # Create output tar with sampled entries
        logger.info(f"Creating output tar: {output_tar}")
        with tarfile.open(output_tar, "w") as dst_tar:
            for i, member in enumerate(sampled_members, 1):
                # Extract file object from source
                file_obj = src_tar.extractfile(member)
                if file_obj is None:
                    logger.warning(f"Skipping {member.name}: unable to extract")
                    continue
                
                # Add to destination tar
                dst_tar.addfile(member, file_obj)
                
                if i % 10 == 0 or i == sample_size:
                    logger.info(f"Progress: {i}/{sample_size} files added")
    
    # Report output size
    output_size_mb = output_tar.stat().st_size / (1024 * 1024)
    logger.info(f"Output tar created: {output_tar} ({output_size_mb:.2f} MB)")
    logger.info(f"Sampling complete: {sample_size} samples extracted")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample random entries from a tar archive (default: 200 samples)"
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source tar archive path (marginalia dump)"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output tar archive path (default: dataset/sample-test-{N}.tar)"
    )
    parser.add_argument(
        "-n", "--sample-size",
        type=int,
        default=200,
        help="Number of samples to extract (default: 200)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = script_dir / f"sample-test-{args.sample_size}.tar"
        logger.info(f"Auto-generated output path: {args.output}")
    
    try:
        sample_tar(
            source_tar=args.source,
            output_tar=args.output,
            sample_size=args.sample_size,
            seed=args.seed
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
