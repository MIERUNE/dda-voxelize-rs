# dda-voxelizer

[![test](https://github.com/MIERUNE/dda-voxelizer-rs/actions/workflows/test.yml/badge.svg)](https://github.com/MIERUNE/dda-voxelizer-rs/actions/workflows/test.yml)
<!-- [![codecov](https://codecov.io/github/MIERUNE/dda-voxelize-rs/graph/badge.svg?token=DZb9Met7wY)](https://codecov.io/github/MIERUNE/dda-voxelize-rs) -->
[![Crates.io Version](https://img.shields.io/crates/v/dda-voxelizer)](https://crates.io/crates/dda-voxelizer)

![1716994116122](image/README/1716994116122.png)

This project is a voxelizer implementation in Rust using the Digital Differential Analyzer (DDA) algorithm. The DDA algorithm was chosen for its simplicity and efficiency in voxelizing 3D models.

## DDA Algorithm Overview

The DDA algorithm is a fast line drawing method commonly used in computer graphics. It incrementally steps along the major axis of the line (X or Y), while computing the corresponding value on the minor axis at each step. This allows lines to be drawn efficiently by only visiting the pixels/voxels that the line actually intersects.

For 3D voxelization, the DDA algorithm can be extended to incrementally step along the dominant axis (X, Y, or Z), while tracking the intersection points on the other two axes. This enables rapid traversal of the voxel grid to tag all voxels overlapping with the geometry.

## Example

Run the example: `cargo run --package dda-voxelize --example voxelize`
