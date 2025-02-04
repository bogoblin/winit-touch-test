struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) relative_to_center: vec2<f32>,
}

struct TouchInstance {
    @location(0) position: vec2<f32>,
    @location(1) force: f32,
    @location(2) scale: f32,
}

struct CameraUniform {
    @location(0) dimensions: vec2<f32>,
    @location(1) scale_factor: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32, instance: TouchInstance) -> VertexOut {
    var relative_to_center: vec2<f32>;
    switch (idx % 6) {
        case 0u: {
            relative_to_center = vec2(-1.0, -1.0);
        }
        case 1u: {
            relative_to_center = vec2(1.0, 1.0);
        }
        case 2u: {
            relative_to_center = vec2(1.0, -1.0);
        }
        case 3u: {
            relative_to_center = vec2(-1.0, -1.0);
        }
        case 4u: {
            relative_to_center = vec2(-1.0, 1.0);
        }
        case 5u: {
            relative_to_center = vec2(1.0, 1.0);
        }
        default: {}
    }

    var out: VertexOut;

    out.position = vec4(screen_to_clip(instance.position + instance.scale * relative_to_center), 0.0, 1.0);
    out.relative_to_center = relative_to_center;

    return out;
}

fn screen_to_clip(screen_coords: vec2<f32>) -> vec2<f32> {
    var clip = ((screen_coords / (camera.dimensions.xy * camera.scale_factor)) * 2.0) - vec2(1.0, 1.0);
    clip.y *= -1.0;
    return clip;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let distance_from_center = length(in.relative_to_center);
    let distance_from_edge = 1.0 - distance_from_center;
    return vec4(0.9, 0.9, 0.9, distance_from_edge*10.0);
}