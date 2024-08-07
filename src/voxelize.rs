use glam::Vec3;
use hashbrown::{hash_map::Entry, HashMap};

pub type VoxelPosition = [i32; 3];
pub type VertexWeight = [f32; 3];

pub trait Shader<V>: Fn(Option<&V>, VoxelPosition, VertexWeight) -> V {}
impl<F, V> Shader<V> for F where F: Fn(Option<&V>, VoxelPosition, VertexWeight) -> V {}

pub struct DdaVoxelizer<V> {
    buffer: HashMap<VoxelPosition, V>,
}

impl<V> DdaVoxelizer<V> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<V> Default for DdaVoxelizer<V> {
    fn default() -> Self {
        Self {
            buffer: HashMap::new(),
        }
    }
}

impl<V: Clone> DdaVoxelizer<V> {
    pub fn add_triangle(&mut self, triangle: &[[f32; 3]; 3], shader: &impl Shader<V>) {
        fill_triangle(&mut self.buffer, triangle, &shader);
    }

    pub fn add_line(&mut self, start: [f32; 3], end: [f32; 3], shader: &impl Shader<V>) {
        draw_line(
            &mut self.buffer,
            Vec3::from_array(start),
            Vec3::from_array(end),
            Vec3::from_array([1., 0., 0.]),
            Vec3::from_array([0., 1., 0.]),
            shader,
        )
    }

    pub fn finalize(self) -> HashMap<VoxelPosition, V> {
        self.buffer
    }
}

fn put_voxel<V: Clone>(
    buffer: &mut HashMap<[i32; 3], V>,
    position: glam::IVec3,
    weight: glam::Vec3,
    shader: impl Shader<V>,
) {
    let position = position.to_array();
    let weight = {
        let mut w = weight.to_array();
        w[0] = w[0].clamp(0.0, 1.0);
        w[1] = w[1].clamp(0.0, 1.0);
        w[2] = w[2].clamp(0.0, 1.0);
        w
    };

    match buffer.entry(position) {
        Entry::Occupied(mut v) => {
            let v = v.get_mut();
            *v = shader(Some(v), position, weight);
        }
        Entry::Vacant(v) => {
            v.insert(shader(None, position, weight));
        }
    }
}

fn draw_line<V: Clone>(
    buffer: &mut HashMap<VoxelPosition, V>,
    start: Vec3,
    end: Vec3,
    start_w: Vec3,
    end_w: Vec3,
    shader: &impl Shader<V>,
) {
    let difference = end - start;
    let mut current_voxel = (start + Vec3::splat(0.5)).floor().as_ivec3();
    let last_voxel = (end + Vec3::splat(0.5)).floor().as_ivec3();

    let step = difference.signum().as_ivec3();
    let next_voxel_boundary = current_voxel.as_vec3() + 0.5 * step.as_vec3();
    let mut tmax = (next_voxel_boundary - start) / difference;
    let tdelta = step.as_vec3() / difference;

    let dir = difference.normalize();
    let distance = difference.length();
    let x_inv = dir.x.abs() / distance;
    let y_inv = dir.y.abs() / distance;
    let z_inv = dir.z.abs() / distance;
    let mut t = 0.0;

    // TODO: We could optimize this since we actually need 2D DDA, not 3D DDA

    while current_voxel != last_voxel && t < 1.0 {
        put_voxel(
            buffer,
            current_voxel,
            start_w + t * (end_w - start_w),
            shader,
        );

        if tmax.x < tmax.y {
            if tmax.x < tmax.z {
                current_voxel.x += step.x;
                tmax.x += tdelta.x;
                t += x_inv;
            } else {
                current_voxel.z += step.z;
                tmax.z += tdelta.z;
                t += z_inv;
            }
        } else if tmax.y < tmax.z {
            current_voxel.y += step.y;
            tmax.y += tdelta.y;
            t += y_inv;
        } else {
            current_voxel.z += step.z;
            tmax.z += tdelta.z;
            t += z_inv;
        }
    }

    if t <= 1.0 {
        put_voxel(
            buffer,
            current_voxel,
            start_w + t * (end_w - start_w),
            shader,
        );
    }
}

fn fill_triangle<V: Clone>(
    voxels: &mut HashMap<VoxelPosition, V>,
    triangle: &[[f32; 3]; 3],
    shader: &impl Shader<V>,
) {
    let v0 = Vec3::from(triangle[0]);
    let v1 = Vec3::from(triangle[1]);
    let v2 = Vec3::from(triangle[2]);

    let mut normal = (v1 - v0).cross(v2 - v0);
    let normal_length = normal.length();
    if normal_length.is_nan() || normal_length == 0.0 {
        // TODO: Should we draw a line when the triangle is colinear?
        return;
    }
    normal /= normal_length;

    let normal_axis = normal
        .abs()
        .to_array()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // Determine the axis to sweep along
    let sweep_axis = {
        let min_point = v0.min(v1).min(v2);
        let max_point = v0.max(v1).max(v2);
        let box_size = max_point - min_point;

        // Which axis is the triangle's normal closer to?
        match normal_axis {
            0 if box_size[1] >= box_size[2] => 1,
            0 => 2,
            1 if box_size[2] >= box_size[0] => 2,
            1 => 0,
            2 if box_size[0] >= box_size[1] => 0,
            2 => 1,
            _ => unreachable!(),
        }
    };

    // Reorder the vertices to be arranged sequentially along the sweep axis
    let mut ordered_verts = [triangle[0], triangle[1], triangle[2]];
    let mut vert_order = [0, 1, 2];
    if ordered_verts[0][sweep_axis] > ordered_verts[1][sweep_axis] {
        ordered_verts.swap(0, 1);
        vert_order.swap(0, 1);
    }
    if ordered_verts[1][sweep_axis] > ordered_verts[2][sweep_axis] {
        ordered_verts.swap(1, 2);
        vert_order.swap(1, 2);
    }
    if ordered_verts[0][sweep_axis] > ordered_verts[1][sweep_axis] {
        ordered_verts.swap(0, 1);
        vert_order.swap(0, 1);
    }
    debug_assert!(ordered_verts[1][sweep_axis] >= ordered_verts[0][sweep_axis]);

    let [mut w0, mut w1, mut w2] = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
    w0[vert_order[0]] = 1.0;
    w1[vert_order[1]] = 1.0;
    w2[vert_order[2]] = 1.0;

    let p0 = Vec3::from(ordered_verts[0]);
    let p1 = Vec3::from(ordered_verts[1]);
    let p2 = Vec3::from(ordered_verts[2]);
    let v01 = p1 - p0;
    let v02 = p2 - p0;
    let v12 = p2 - p1;

    let end_step = v02 / (ordered_verts[2][sweep_axis] - ordered_verts[0][sweep_axis]);
    let mut end_pos = p0
        + end_step
            * ((1.0 - ordered_verts[0][sweep_axis] + ordered_verts[0][sweep_axis].floor()) % 1.0);
    let end_w_step = (w2 - w0) / (ordered_verts[2][sweep_axis] - ordered_verts[0][sweep_axis]);
    let mut end_w = w0
        + end_w_step
            * ((1.0 - ordered_verts[0][sweep_axis] + ordered_verts[0][sweep_axis].floor()) % 1.0);

    let start_step1 = v01 / (ordered_verts[1][sweep_axis] - ordered_verts[0][sweep_axis]);
    let start_step2 = v12 / (ordered_verts[2][sweep_axis] - ordered_verts[1][sweep_axis]);
    let start_w_step1 = (w1 - w0) / (ordered_verts[1][sweep_axis] - ordered_verts[0][sweep_axis]);
    let start_w_step2 = (w2 - w1) / (ordered_verts[2][sweep_axis] - ordered_verts[1][sweep_axis]);
    let to_next_line =
        (1.0 - ordered_verts[0][sweep_axis] + ordered_verts[0][sweep_axis].floor()) % 1.0;
    let mut start_pos = p0 + start_step1 * to_next_line;
    let mut start_w = w0 + start_w_step1 * to_next_line;

    if start_step1[sweep_axis].is_finite() {
        // underjet
        if to_next_line > 0.5 {
            let d = to_next_line - 0.5;
            let mut s = (p0 + d * v01).to_array();
            let mut e = (p0 + d * v02).to_array();
            s[sweep_axis] -= 0.5;
            e[sweep_axis] -= 0.5;
            draw_line(
                voxels,
                Vec3::from_array(s),
                Vec3::from_array(e),
                w0 + d * (w1 - w0),
                w0 + d * (w2 - w0),
                shader,
            );
        };

        // Start position is on the first edge
        while start_pos[sweep_axis] <= ordered_verts[1][sweep_axis] {
            draw_line(voxels, start_pos, end_pos, start_w, end_w, shader);
            start_pos += start_step1;
            end_pos += end_step;
            start_w += start_w_step1;
            end_w += end_w_step;
        }

        // Switch to the second edge
        let end_vertex_y = ordered_verts[1][sweep_axis] - start_pos[sweep_axis];
        start_pos += (start_step1 - start_step2) * end_vertex_y;
        start_w += (start_w_step1 - start_w_step2) * end_vertex_y;
    } else {
        // Switch to the second edge
        start_pos = p1
            + start_step2
                * ((1.0 - ordered_verts[1][sweep_axis] + ordered_verts[1][sweep_axis].floor())
                    % 1.0);
        start_w = w1
            + start_w_step2
                * ((1.0 - ordered_verts[1][sweep_axis] + ordered_verts[1][sweep_axis].floor())
                    % 1.0);
    }

    if start_step2[sweep_axis].is_finite() {
        // Start position is on the second edge
        while end_pos[sweep_axis] <= ordered_verts[2][sweep_axis] {
            draw_line(voxels, start_pos, end_pos, start_w, end_w, shader);
            start_pos += start_step2;
            end_pos += end_step;
            start_w += start_w_step2;
            end_w += end_w_step;
        }

        // overjet
        if end_pos[sweep_axis] - 0.5 < ordered_verts[2][sweep_axis]
            && end_pos[sweep_axis] - 0.5 > ordered_verts[1][sweep_axis]
        {
            start_pos -= start_step2 * 0.5;
            end_pos -= end_step * 0.5;
            start_w -= start_w_step2 * 0.5;
            end_w -= end_w_step * 0.5;

            start_pos[sweep_axis] += 0.5;
            end_pos[sweep_axis] += 0.5;

            draw_line(voxels, start_pos, end_pos, start_w, end_w, shader);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use earcut::{utils3d::project3d_to_2d, Earcut};
    use flatgeom::MultiPolygon;

    #[test]
    fn test_minimum_polygon() {
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [0.0, 0.1, 0.0],
        ];

        let mut mpoly = MultiPolygon::<u32>::new();

        mpoly.add_exterior([0, 1, 2, 3, 0]);

        let mut earcutter = Earcut::new();
        let mut buf3d: Vec<[f32; 3]> = Vec::new();
        let mut buf2d: Vec<[f32; 2]> = Vec::new();
        let mut index_buf: Vec<u32> = Vec::new();

        let mut voxelizer = DdaVoxelizer {
            buffer: HashMap::new(),
        };

        for idx_poly in mpoly.iter() {
            let poly = idx_poly.transform(|idx| vertices[*idx as usize]);
            let num_outer = match poly.hole_indices().first() {
                Some(&v) => v as usize,
                None => poly.raw_coords().len(),
            };

            buf3d.clear();
            buf3d.extend(
                poly.raw_coords()
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]),
            );

            if project3d_to_2d(&buf3d, num_outer, &mut buf2d) {
                earcutter.earcut(buf2d.iter().cloned(), poly.hole_indices(), &mut index_buf);
                for index in index_buf.chunks_exact(3) {
                    voxelizer.add_triangle(
                        &[
                            buf3d[index[0] as usize],
                            buf3d[index[1] as usize],
                            buf3d[index[2] as usize],
                        ],
                        &|_, _, _| true,
                    );
                }
            }
        }

        let occupied_voxels = voxelizer.finalize();
        let mut test_voxels: HashMap<VoxelPosition, bool> = HashMap::new();
        test_voxels.insert([0, 0, 0], true);

        assert_eq!(occupied_voxels, test_voxels);
    }

    #[test]
    fn test_square_polygon() {
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let mut mpoly = MultiPolygon::<u32>::new();

        mpoly.add_exterior([0, 1, 2, 3, 0]);

        let mut earcutter = Earcut::new();
        let mut buf3d: Vec<[f32; 3]> = Vec::new();
        let mut buf2d: Vec<[f32; 2]> = Vec::new();
        let mut index_buf: Vec<u32> = Vec::new();

        let mut voxelizer = DdaVoxelizer {
            buffer: HashMap::new(),
        };
        for idx_poly in mpoly.iter() {
            let poly = idx_poly.transform(|idx| vertices[*idx as usize]);
            let num_outer = match poly.hole_indices().first() {
                Some(&v) => v as usize,
                None => poly.raw_coords().len(),
            };

            buf3d.clear();
            buf3d.extend(
                poly.raw_coords()
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]),
            );

            if project3d_to_2d(&buf3d, num_outer, &mut buf2d) {
                earcutter.earcut(buf2d.iter().cloned(), poly.hole_indices(), &mut index_buf);
                for indx in index_buf.chunks_exact(3) {
                    voxelizer.add_triangle(
                        &[
                            buf3d[indx[0] as usize],
                            buf3d[indx[1] as usize],
                            buf3d[indx[2] as usize],
                        ],
                        &|_, _, _| true,
                    );
                }
            }
        }

        let occupied_voxels = voxelizer.finalize();

        let mut test_voxels: HashMap<VoxelPosition, bool> = HashMap::new();
        test_voxels.insert([0, 0, 0], true);
        test_voxels.insert([1, 0, 0], true);
        test_voxels.insert([0, 1, 0], true);
        test_voxels.insert([1, 1, 0], true);

        assert_eq!(occupied_voxels, test_voxels);
    }

    #[test]
    fn test_hole_polygon() {
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.49, 0.49, 0.0],
            [2.51, 0.49, 0.0],
            [2.51, 2.51, 0.0],
            [0.49, 2.51, 0.0],
        ];

        let mut mpoly = MultiPolygon::<u32>::new();

        mpoly.add_exterior([0, 1, 2, 3, 0]);
        mpoly.add_interior([4, 5, 6, 7, 4]);

        let mut earcutter = Earcut::new();
        let mut buf3d: Vec<[f32; 3]> = Vec::new();
        let mut buf2d: Vec<[f32; 2]> = Vec::new();
        let mut index_buf: Vec<u32> = Vec::new();

        let mut voxelizer = DdaVoxelizer {
            buffer: HashMap::new(),
        };

        for idx_poly in mpoly.iter() {
            let poly = idx_poly.transform(|idx| vertices[*idx as usize]);
            let num_outer = match poly.hole_indices().first() {
                Some(&v) => v as usize,
                None => poly.raw_coords().len(),
            };

            buf3d.clear();
            buf3d.extend(
                poly.raw_coords()
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]),
            );

            if project3d_to_2d(&buf3d, num_outer, &mut buf2d) {
                earcutter.earcut(buf2d.iter().cloned(), poly.hole_indices(), &mut index_buf);
                for indx in index_buf.chunks_exact(3) {
                    voxelizer.add_triangle(
                        &[
                            buf3d[indx[0] as usize],
                            buf3d[indx[1] as usize],
                            buf3d[indx[2] as usize],
                        ],
                        &|_, _, _| true,
                    );
                }
            }
        }

        let occupied_voxels = voxelizer.finalize();

        let mut test_voxels: HashMap<VoxelPosition, bool> = HashMap::new();
        for p in [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [0, 1, 0],
            [3, 1, 0],
            [0, 2, 0],
            [3, 2, 0],
            [0, 3, 0],
            [1, 3, 0],
            [2, 3, 0],
            [3, 3, 0],
        ] {
            test_voxels.insert(p, true);
        }

        assert_eq!(occupied_voxels, test_voxels);
    }

    #[test]
    fn test_cube() {
        let vertices: Vec<[f64; 3]> = vec![
            // exterior
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0],
            [0.0, 10.0, 10.0],
            // interior
            [3.0, 3.0, 0.0],
            [7.0, 3.0, 0.0],
            [7.0, 7.0, 0.0],
            [3.0, 7.0, 0.0],
            [3.0, 3.0, 10.0],
            [7.0, 3.0, 10.0],
            [7.0, 7.0, 10.0],
            [3.0, 7.0, 10.0],
        ];

        let mut mpoly = MultiPolygon::<u32>::new();

        // index
        // 1st polygon
        mpoly.add_exterior([0, 1, 2, 3, 0]);
        mpoly.add_interior([8, 9, 10, 11, 8]);
        // 2nd polygon
        mpoly.add_exterior([4, 5, 6, 7, 4]);
        mpoly.add_interior([12, 13, 14, 15, 12]);
        // 3rd polygon
        mpoly.add_exterior([0, 1, 5, 4, 0]);
        // 4th polygon
        mpoly.add_exterior([1, 2, 6, 5, 1]);
        // 6th polygon
        mpoly.add_exterior([2, 3, 7, 6, 2]);
        // 6th polygon
        mpoly.add_exterior([3, 0, 4, 7, 3]);

        let mut earcutter = Earcut::new();
        let mut buf3d: Vec<[f32; 3]> = Vec::new();
        let mut buf2d: Vec<[f32; 2]> = Vec::new();
        let mut index_buf: Vec<u32> = Vec::new();

        let mut voxelizer = DdaVoxelizer {
            buffer: HashMap::new(),
        };

        for idx_poly in mpoly.iter() {
            let poly = idx_poly.transform(|idx| vertices[*idx as usize]);
            let num_outer = match poly.hole_indices().first() {
                Some(&v) => v as usize,
                None => poly.raw_coords().len(),
            };

            buf3d.clear();
            buf3d.extend(
                poly.raw_coords()
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]),
            );

            if project3d_to_2d(&buf3d, num_outer, &mut buf2d) {
                earcutter.earcut(buf2d.iter().cloned(), poly.hole_indices(), &mut index_buf);
                for indx in index_buf.chunks_exact(3) {
                    voxelizer.add_triangle(
                        &[
                            buf3d[indx[0] as usize],
                            buf3d[indx[1] as usize],
                            buf3d[indx[2] as usize],
                        ],
                        &|_, _, _| true,
                    );
                }
            }
        }

        let occupied_voxels = voxelizer.finalize();
        assert_eq!(occupied_voxels.len(), 584);
    }
}
