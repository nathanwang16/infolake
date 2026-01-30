"""
Visualizer module for Truth Atlas.

Provides web UI and API server for exploring the atlas:
- REST API for querying documents and mappings
- Interactive 2D/3D map visualization
- Document detail views and search
"""

from visualizer.server import AtlasServer

__all__ = ['AtlasServer']
