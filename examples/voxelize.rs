use std::{
    fs::File,
    io::{BufWriter, Write},
};

use byteorder::{LittleEndian, WriteBytesExt};
use earcut::{utils3d::project3d_to_2d, Earcut};
use flatgeom::MultiPolygon;
use indexmap::IndexSet;
use palette::FromColor;
use serde_json::json;

use dda_voxelize::DdaVoxelizer;

fn main() {
    let vertices: Vec<[f64; 3]> = vec![
        // exterior
        [26.5, 0.5, 0.5],
        [0.5, 0.5, 26.5],
        [-25.5, 0.5, 0.5],
        [0.5, 0.5, -25.5],
        [0.5, 45.5, 0.5],
        [0.5, -44.5, 0.5],
    ];

    let mut mpoly = MultiPolygon::<u32>::new();

    // index
    mpoly.add_exterior([0, 1, 2, 3]);
    mpoly.add_exterior([0, 1, 4]);
    mpoly.add_exterior([1, 2, 4]);
    mpoly.add_exterior([2, 3, 4]);
    mpoly.add_exterior([3, 0, 4]);
    mpoly.add_exterior([0, 1, 5]);
    mpoly.add_exterior([1, 2, 5]);
    mpoly.add_exterior([2, 3, 5]);
    mpoly.add_exterior([3, 0, 5]);

    // triangulation
    let mut earcutter = Earcut::new();
    let mut buf3d: Vec<[f32; 3]> = Vec::new();
    let mut buf2d: Vec<[f32; 2]> = Vec::new();
    let mut index_buf: Vec<u32> = Vec::new();

    let mut voxelizer = DdaVoxelizer::new();

    for idx_poly in mpoly.iter() {
        let poly = idx_poly.transform(|idx| vertices[*idx as usize]);
        let num_outer = match poly.hole_indices().first() {
            Some(&v) => v as usize,
            None => poly.raw_coords().len(),
        };

        for axis in 0..=2 {
            buf3d.clear();
            buf3d.extend(poly.raw_coords().iter().map(|v| match axis {
                0 => [v[0] as f32, v[1] as f32, v[2] as f32],
                1 => [v[1] as f32, v[0] as f32, v[2] as f32],
                2 => [v[0] as f32, v[2] as f32, v[1] as f32],
                _ => unreachable!(),
            }));
            if project3d_to_2d(&buf3d, num_outer, &mut buf2d) {
                // earcut
                earcutter.earcut(buf2d.iter().cloned(), poly.hole_indices(), &mut index_buf);
                for index in index_buf.chunks_exact(3) {
                    voxelizer.add_triangle(
                        &[
                            buf3d[index[0] as usize],
                            buf3d[index[1] as usize],
                            buf3d[index[2] as usize],
                        ],
                        &|_current_value, [x, y, z], _vertex_weight| {
                            let [x, y, z] = [x as f32, y as f32, z as f32];
                            let color_lab = palette::Okhsl::new(
                                x.atan2(z).to_degrees(),
                                1.0 - (x * x + z * z) / 2200.,
                                y / 90. + 0.5,
                            );
                            let color_srgb = palette::Srgb::from_color(color_lab);
                            [color_srgb.red, color_srgb.green, color_srgb.blue]
                        },
                    );
                }
            }
        }
    }

    voxelizer.add_triangle(
        &[[40., 40., 40.], [40., 40.6, 40.], [40., 40., 40.6]],
        &|_, [x, y, z], _| {
            let [x, y, z] = [x as f32, y as f32, z as f32];
            let color_lab = palette::Okhsl::new(
                x.atan2(z).to_degrees(),
                1.0 - (x * x + z * z) / 2200.,
                y / 90. + 0.5,
            );
            let color_srgb = palette::Srgb::from_color(color_lab);
            [color_srgb.red, color_srgb.green, color_srgb.blue]
        },
    );

    // voxelizer.add_line([40., 40., 40.], [40., 40., 40.], &|_, [x, y, z], _| {
    //     let [x, y, z] = [x as f32, y as f32, z as f32];
    //     let color_lab = palette::Okhsl::new(
    //         x.atan2(z).to_degrees(),
    //         1.0 - (x * x + z * z) / 2200.,
    //         y / 90. + 0.5,
    //     );
    //     let color_srgb = palette::Srgb::from_color(color_lab);
    //     [color_srgb.red, color_srgb.green, color_srgb.blue]
    // });

    let occupied_voxels = voxelizer.finalize();

    // -------------------make glTF-------------------

    // voxel is an integer value, but componentType of accessors is 5126 (floating point number),
    // and INTEGER type cannot be used due to primitives constraints

    let mut indices = Vec::new();
    let mut vertices = IndexSet::new(); // [x, y, z, r, g, b]

    for (position, voxel) in occupied_voxels.iter() {
        let [x, y, z] = [position[0] as f32, position[1] as f32, position[2] as f32];
        let [r, g, b] = voxel;

        let [r_bits, g_bits, b_bits] = [r.to_bits(), g.to_bits(), b.to_bits()];

        // Make a voxel cube
        let (idx0, _) = vertices.insert_full([
            (x + 0.5).to_bits(),
            (y - 0.5).to_bits(),
            (z + 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx1, _) = vertices.insert_full([
            (x - 0.5).to_bits(),
            (y - 0.5).to_bits(),
            (z + 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx2, _) = vertices.insert_full([
            (x + 0.5).to_bits(),
            (y - 0.5).to_bits(),
            (z - 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx3, _) = vertices.insert_full([
            (x - 0.5).to_bits(),
            (y - 0.5).to_bits(),
            (z - 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx4, _) = vertices.insert_full([
            (x + 0.5).to_bits(),
            (y + 0.5).to_bits(),
            (z + 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx5, _) = vertices.insert_full([
            (x - 0.5).to_bits(),
            (y + 0.5).to_bits(),
            (z + 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx6, _) = vertices.insert_full([
            (x + 0.5).to_bits(),
            (y + 0.5).to_bits(),
            (z - 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        let (idx7, _) = vertices.insert_full([
            (x - 0.5).to_bits(),
            (y + 0.5).to_bits(),
            (z - 0.5).to_bits(),
            r_bits,
            g_bits,
            b_bits,
        ]);
        indices.extend(
            [
                idx0, idx1, idx2, idx2, idx1, idx3, idx6, idx5, idx4, idx5, idx6, idx7, idx2, idx3,
                idx6, idx7, idx6, idx3, idx4, idx1, idx0, idx1, idx4, idx5, idx0, idx2, idx4, idx6,
                idx4, idx2, idx5, idx3, idx1, idx3, idx5, idx7,
            ]
            .iter()
            .map(|&idx| idx as u32),
        );
    }

    let mut min_position = [f32::MAX; 3];
    let mut max_position = [f32::MIN; 3];
    {
        let mut bin_file = BufWriter::new(File::create("output.bin").unwrap());

        for &idx in &indices {
            bin_file.write_u32::<LittleEndian>(idx).unwrap();
        }

        for &[x, y, z, r, g, b] in &vertices {
            min_position[0] = f32::min(min_position[0], f32::from_bits(x));
            min_position[1] = f32::min(min_position[1], f32::from_bits(y));
            min_position[2] = f32::min(min_position[2], f32::from_bits(z));
            max_position[0] = f32::max(max_position[0], f32::from_bits(x));
            max_position[1] = f32::max(max_position[1], f32::from_bits(y));
            max_position[2] = f32::max(max_position[2], f32::from_bits(z));

            bin_file.write_u32::<LittleEndian>(x).unwrap();
            bin_file.write_u32::<LittleEndian>(y).unwrap();
            bin_file.write_u32::<LittleEndian>(z).unwrap();
            bin_file.write_u32::<LittleEndian>(r).unwrap();
            bin_file.write_u32::<LittleEndian>(g).unwrap();
            bin_file.write_u32::<LittleEndian>(b).unwrap();
        }
    }

    let indices_size = indices.len() * 4;
    let vertices_size = vertices.len() * 6 * 4;
    let total_size = indices_size + vertices_size;

    // make glTF
    let gltf_json = json!( {
        "asset": {
            "version": "2.0",
        },
        "scene": 0,
        "scenes": [
            {
                "nodes": [0],
            },
        ],
        "nodes": [
            {"mesh": 0},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 1,
                            "COLOR_0": 2,
                        },
                        "indices": 0,
                        "mode": 4, // TRIANGLES
                    },
                ],
            },
        ],
        "buffers": [
            {
                "uri": "./output.bin",
                "byteLength": total_size,
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": indices_size,
                "target": 34963, // ELEMENT_ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteStride": 6 * 4,
                "byteOffset": indices_size,
                "byteLength": vertices_size,
                "target": 34962, // ARRAY_BUFFER
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5125, // UNSIGNED_INT
                "count": indices.len(),
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126, // FLOAT
                "count": vertices.len(),
                "type": "VEC3",
                "min": [min_position[0], min_position[1], min_position[2]],
                "max": [max_position[0], max_position[1], max_position[2]],
            },
            {
                "bufferView": 1,
                "byteOffset": 4 * 3,
                "componentType": 5126, // FLOAT
                "count": vertices.len(),
                "type": "VEC3",
            },
        ],
    });

    // write glTF
    println!("write glTF");
    let mut gltf_file = File::create("output.gltf").unwrap();
    let _ = gltf_file.write_all(gltf_json.to_string().as_bytes());
}
