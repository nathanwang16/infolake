"""
Dump format adapters for various data sources.

Supported formats:
- SLOP (Marginalia Search) - .slop.zip inside tar archives
- JSONL - Lines of JSON with 'url' field
- Plain text - One URL per line

SLOP format specification:
    Columnar storage with self-documenting filenames encoding schema, type, compression.
    Key files: url.0.dat.s8[].zstd (URLs), body.0.dat.s8[].zstd (HTML content)
    Length files: *.dat-len.varint.bin (variable-length encoded record lengths)
"""

import io
import json
import struct
import tarfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None

from atlas_core import get_logger

logger = get_logger("dump_adapters")


@dataclass
class DumpRecord:
    """Represents a single record from a dump file."""
    url: str
    html: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None


class DumpAdapter(ABC):
    """Base class for dump format adapters."""
    
    @abstractmethod
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Yields DumpRecords from the dump file."""
        raise NotImplementedError("Subclasses must implement iterate()")
    
    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Returns True if this adapter can handle the given file."""
        raise NotImplementedError("Subclasses must implement supports()")


class SlopAdapter(DumpAdapter):
    """
    Adapter for Marginalia SLOP format (columnar storage in .slop.zip files).
    
    SLOP stores data in columns with filenames encoding the schema:
    - url.0.dat.s8[].zstd: URLs as concatenated strings, zstd compressed
    - body.0.dat.s8[].zstd: HTML bodies as concatenated strings
    - *.dat-len.varint.bin: Variable-length encoded record lengths
    """
    
    def supports(self, path: Path) -> bool:
        """Supports tar files containing .slop.zip entries."""
        path_str = str(path)
        return path_str.endswith('.tar') or path_str.endswith('.tar.gz') or path_str.endswith('.tgz')
    
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Iterates through all SLOP zip files in the tar archive."""
        if not path.exists():
            raise FileNotFoundError(f"Dump file not found: {path}")
        
        mode = "r:gz" if str(path).endswith(('.tar.gz', '.tgz')) else "r:"
        
        try:
            with tarfile.open(path, mode=mode) as tar:
                members = [m for m in tar.getmembers() if m.name.endswith('.slop.zip')]
                total_members = len(members)
                
                logger.info(f"Found {total_members} SLOP archives in {path}")
                
                for idx, member in enumerate(members):
                    if not member.isfile():
                        continue
                    
                    try:
                        slop_file = tar.extractfile(member)
                        if slop_file:
                            slop_bytes = slop_file.read()
                            yield from self._parse_slop_zip(slop_bytes, member.name)
                            
                            if (idx + 1) % 100 == 0:
                                logger.info(f"Processed {idx + 1}/{total_members} SLOP archives")
                                
                    except Exception as e:
                        logger.error(f"Failed to process SLOP archive {member.name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open tar archive {path}: {e}")
            raise
    
    def _parse_slop_zip(self, zip_bytes: bytes, archive_name: str) -> Generator[DumpRecord, None, None]:
        """Parses a single .slop.zip file and yields records."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
                namelist = zf.namelist()
                
                # Find required files using pattern matching
                url_data_file = next((n for n in namelist if 'url' in n and n.endswith('.zstd')), None)
                url_len_file = next((n for n in namelist if 'url' in n and 'len' in n and n.endswith('.bin')), None)
                body_data_file = next((n for n in namelist if 'body' in n and n.endswith('.zstd')), None)
                body_len_file = next((n for n in namelist if 'body' in n and 'len' in n and n.endswith('.bin')), None)
                domain_file = next((n for n in namelist if 'domain' in n and 'dic' in n and n.endswith('.bin')), None)
                
                if not url_data_file or not url_len_file:
                    logger.warning(f"Missing URL data in {archive_name}")
                    return
                
                # Read and decompress URL data
                if not ZSTD_AVAILABLE:
                    logger.error("zstandard library not installed. Install with: pip install zstandard")
                    return
                
                url_data_compressed = zf.read(url_data_file)
                url_lengths_raw = zf.read(url_len_file)
                
                # Decompress URL data using streaming decompressor (handles frames without content size)
                url_data = self._decompress_zstd(url_data_compressed)
                
                # Parse varint-encoded lengths
                url_lengths = self._parse_varints(url_lengths_raw)
                
                # Extract individual URLs
                urls = self._extract_strings(url_data, url_lengths)
                
                # Parse body data if available
                bodies = []
                if body_data_file and body_len_file:
                    try:
                        body_data_compressed = zf.read(body_data_file)
                        body_lengths_raw = zf.read(body_len_file)
                        
                        # Decompress body data
                        body_data = self._decompress_zstd(body_data_compressed)
                        
                        body_lengths = self._parse_varints(body_lengths_raw)
                        bodies = self._extract_strings(body_data, body_lengths)
                    except Exception as e:
                        logger.warning(f"Failed to parse body data in {archive_name}: {e}")
                
                # Parse domain if available
                domain = None
                if domain_file:
                    try:
                        domain_raw = zf.read(domain_file)
                        domain = domain_raw.decode('utf-8', errors='ignore').strip('\x00')
                    except Exception:
                        pass
                
                # Yield records
                for i, url in enumerate(urls):
                    if not url or not url.startswith('http'):
                        continue
                    
                    html = bodies[i] if i < len(bodies) else None
                    yield DumpRecord(
                        url=url,
                        html=html,
                        domain=domain,
                        metadata={'source_archive': archive_name}
                    )
                    
        except Exception as e:
            logger.error(f"Failed to parse SLOP zip {archive_name}: {e}")
    
    def _decompress_zstd(self, compressed_data: bytes) -> bytes:
        """
        Decompresses zstd data, handling both standard and streaming formats.
        
        Streaming zstd frames don't include content size in header, which
        causes the standard decompress() to fail. We use stream_reader() as fallback.
        """
        if not ZSTD_AVAILABLE or zstd is None:
            raise RuntimeError("zstandard library not available")
        
        dctx = zstd.ZstdDecompressor()
        
        try:
            # Try standard decompression first
            return dctx.decompress(compressed_data)
        except zstd.ZstdError:
            # Fallback to streaming decompression
            reader = dctx.stream_reader(io.BytesIO(compressed_data))
            result = reader.read()
            reader.close()
            return result
    
    def _parse_varints(self, data: bytes) -> List[int]:
        """
        Parses variable-length encoded integers (varint).
        
        Varint encoding: Each byte's MSB indicates continuation.
        Lower 7 bits contain the value, LSB first.
        """
        lengths = []
        i = 0
        while i < len(data):
            value = 0
            shift = 0
            while i < len(data):
                byte = data[i]
                i += 1
                value |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            lengths.append(value)
        return lengths
    
    def _extract_strings(self, data: bytes, lengths: List[int]) -> List[str]:
        """Extracts strings from concatenated data using length information."""
        strings = []
        offset = 0
        for length in lengths:
            if offset + length > len(data):
                break
            string_bytes = data[offset:offset + length]
            try:
                strings.append(string_bytes.decode('utf-8', errors='ignore'))
            except Exception:
                strings.append('')
            offset += length
        return strings


class JSONLAdapter(DumpAdapter):
    """Adapter for JSONL (JSON Lines) format dumps."""
    
    def supports(self, path: Path) -> bool:
        """Supports .jsonl, .json, and gzipped variants."""
        path_str = str(path).lower()
        return any(path_str.endswith(ext) for ext in ['.jsonl', '.jsonl.gz', '.json', '.json.gz', '.ndjson'])
    
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Iterates through JSONL records."""
        if not path.exists():
            raise FileNotFoundError(f"Dump file not found: {path}")
        
        import gzip
        
        opener = gzip.open if str(path).endswith('.gz') else open
        mode = 'rt'
        
        try:
            with opener(path, mode, encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        url = data.get('url') or data.get('URL') or data.get('link')
                        
                        if not url:
                            continue
                        
                        yield DumpRecord(
                            url=url,
                            html=data.get('html') or data.get('content'),
                            metadata=data.get('metadata', {}),
                            domain=data.get('domain')
                        )
                        
                        if line_num % 10000 == 0:
                            logger.info(f"Processed {line_num} JSONL lines")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to read JSONL file {path}: {e}")
            raise


class PlainTextAdapter(DumpAdapter):
    """Adapter for plain text URL lists (one URL per line)."""
    
    def supports(self, path: Path) -> bool:
        """Supports .txt, .urls, and gzipped variants."""
        path_str = str(path).lower()
        return any(path_str.endswith(ext) for ext in ['.txt', '.txt.gz', '.urls', '.urls.gz'])
    
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Iterates through plain text URLs."""
        if not path.exists():
            raise FileNotFoundError(f"Dump file not found: {path}")
        
        import gzip
        
        opener = gzip.open if str(path).endswith('.gz') else open
        mode = 'rt'
        
        try:
            with opener(path, mode, encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    url = line.strip()
                    
                    if not url or url.startswith('#'):
                        continue
                    
                    # Handle CSV-style lines (take first column)
                    if ',' in url:
                        url = url.split(',')[0].strip()
                    
                    if not url.startswith('http'):
                        continue
                    
                    yield DumpRecord(url=url)
                    
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} URLs")
                        
        except Exception as e:
            logger.error(f"Failed to read text file {path}: {e}")
            raise


class CurlieAdapter(DumpAdapter):
    """Adapter for Curlie directory data (DMOZ-style)."""
    
    def supports(self, path: Path) -> bool:
        """Supports curlie data files."""
        return 'curlie' in str(path).lower()
    
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Iterates through Curlie records."""
        if not path.exists():
            raise FileNotFoundError(f"Dump file not found: {path}")
        
        # Curlie data is typically RDF/XML or structured text
        # Simplified implementation for common formats
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Check if it's a download page (not actual data)
                if '<html' in content.lower()[:1000]:
                    logger.warning(f"Curlie file appears to be HTML (possibly a download page): {path}")
                    # Try to extract any URLs from the page
                    import re
                    urls = re.findall(r'href=["\']?(https?://[^"\'\s>]+)', content)
                    for url in set(urls):
                        if 'curlie.org' not in url.lower():
                            yield DumpRecord(url=url)
                    return
                
                # Parse actual Curlie data (RDF or tab-separated)
                import re
                urls = re.findall(r'<link>([^<]+)</link>|<ExternalPage about="([^"]+)"', content)
                
                for match in urls:
                    url = match[0] or match[1]
                    if url and url.startswith('http'):
                        yield DumpRecord(url=url)
                        
        except Exception as e:
            logger.error(f"Failed to read Curlie file {path}: {e}")
            raise


class C4DirectoryAdapter(DumpAdapter):
    """
    Adapter for C4-style directories containing multiple JSONL files.
    
    C4 (Colossal Clean Crawled Corpus) stores data as directories of
    gzipped JSONL files with fields: text, timestamp, url.
    """
    
    def supports(self, path: Path) -> bool:
        """Supports directories containing .json.gz files."""
        if not path.is_dir():
            return False
        # Check if directory contains json.gz files
        json_files = list(path.glob("**/*.json.gz"))
        return len(json_files) > 0
    
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Iterates through all JSONL files in the directory."""
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        import gzip
        
        # Find all json.gz files recursively
        json_files = sorted(path.glob("**/*.json.gz"))
        total_files = len(json_files)
        
        logger.info(f"Found {total_files} JSONL files in {path}")
        
        for file_idx, json_file in enumerate(json_files):
            try:
                with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            url = data.get('url') or data.get('URL')
                            
                            if not url:
                                continue
                            
                            # C4 has pre-extracted text, use it directly
                            text = data.get('text', '')
                            
                            yield DumpRecord(
                                url=url,
                                html=None,  # C4 has text, not HTML
                                metadata={
                                    'text': text,
                                    'timestamp': data.get('timestamp'),
                                    'source_file': json_file.name,
                                    'pre_extracted': True
                                },
                                domain=None
                            )
                            
                        except json.JSONDecodeError:
                            continue
                
                if (file_idx + 1) % 10 == 0:
                    logger.info(f"Processed {file_idx + 1}/{total_files} C4 files")
                    
            except Exception as e:
                logger.error(f"Failed to process C4 file {json_file}: {e}")
                continue


class AdapterRegistry:
    """Registry for dump format adapters with automatic format detection."""
    
    def __init__(self):
        self._adapters: List[DumpAdapter] = [
            C4DirectoryAdapter(),  # Check directory adapters first
            SlopAdapter(),
            JSONLAdapter(),
            PlainTextAdapter(),
            CurlieAdapter(),
        ]
    
    def get_adapter(self, path: Path) -> Optional[DumpAdapter]:
        """Returns the appropriate adapter for the given file."""
        for adapter in self._adapters:
            if adapter.supports(path):
                logger.info(f"Using {adapter.__class__.__name__} for {path}")
                return adapter
        
        logger.error(f"No adapter found for file: {path}")
        return None
    
    def iterate(self, path: Path) -> Generator[DumpRecord, None, None]:
        """Auto-detects format and iterates through records."""
        adapter = self.get_adapter(path)
        if adapter is None:
            raise ValueError(f"Unsupported dump format: {path}")
        yield from adapter.iterate(path)


# Module-level singleton
adapter_registry = AdapterRegistry()


def parse_dump(path: str) -> Generator[DumpRecord, None, None]:
    """
    Convenience function to parse a dump file.
    
    Args:
        path: Path to the dump file
        
    Yields:
        DumpRecord objects for each valid entry
    """
    yield from adapter_registry.iterate(Path(path))
