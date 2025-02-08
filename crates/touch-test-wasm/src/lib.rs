use std::collections::HashMap;
use std::mem;
use cgmath::Vector2;
use wasm_bindgen::prelude::wasm_bindgen;
use wgpu::{include_wgsl, BufferAddress, Color, CompositeAlphaMode, PresentMode, VertexBufferLayout};
use wgpu::VertexFormat::{Float32, Float32x2};
use winit::dpi::{PhysicalPosition};
use winit::event::{ElementState, Force, KeyEvent, Touch, TouchPhase, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes};
use log::info;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::ser::SerializeMap;

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scale_factor: f64,
    window: &'a Window,
    render_pipeline: wgpu::RenderPipeline,
}

#[derive(Debug)]
struct TouchBox(Touch);

impl Serialize for TouchBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("x", &self.0.location.x)?;
        map.serialize_entry("y", &self.0.location.y)?;
        map.serialize_entry("id", &self.0.id)?;
        map.end()
    }
}

#[derive(Serialize, Debug)]
struct Fingers {
    fingers: HashMap<u64, TouchBox>
}

impl Fingers {
    pub(crate) fn remove(&mut self, id: &u64) {
        self.fingers.remove(id);
    }
    pub(crate) fn insert(&mut self, id: u64, finger: Touch) {
        self.fingers.insert(id, TouchBox(finger));
    }
    fn new() -> Self {
        Self {
            fingers: Default::default(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().unwrap();
    let event_loop = EventLoop::new().unwrap();
    let window = event_loop.create_window(
        WindowAttributes::new()
            .with_inner_size(PhysicalSize::new(1280, 720))
    ).unwrap();

    use winit::dpi::PhysicalSize;
    let _ = window.request_inner_size(PhysicalSize::new(1024, 1024));

    use winit::platform::web::WindowExtWebSys;
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let dst = doc.get_element_by_id("wasm-example")?;
            let canvas = web_sys::Element::from(window.canvas()?);
            dst.append_child(&canvas).ok()?;
            Some(())
        })
        .expect("Couldn't append canvas to document body.");

    let mut surface_configured = false;
    let mut fingers: Fingers = Fingers::new();

    let mut size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..Default::default()
    });
    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        },
    ).await.unwrap();

    let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults();
    // required_limits.max_texture_dimension_2d = 4096;
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits,
            label: None,
            memory_hints: Default::default(),
        },
        None,
    ).await.unwrap();

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats.iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: PresentMode::AutoVsync,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 1,
    };

    let camera_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: mem::size_of::<CameraInstance>() as BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
    });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &camera_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }
        ],
    });

    let shader = device.create_shader_module(include_wgsl!("touch.wgsl"));
    let render_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[TouchInstance::desc()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    const MAX_FINGERS: u32 = 10;
    let instance_buffer = device.create_buffer(
        &wgpu::BufferDescriptor {
            label: None,
            size: (MAX_FINGERS as usize * mem::size_of::<TouchInstance>()) as BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );

    let mut state = State {
        surface,
        device,
        queue,
        config,
        scale_factor: 1.0,
        window: &window,
        render_pipeline,
    };

    event_loop.run(move |event, control_flow| {
        match event {
            winit::event::Event::WindowEvent {
                ref event,
                ..
            } => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                        ..
                    } => control_flow.exit(),
                    WindowEvent::Resized(..) => {
                        surface_configured = true;
                        let win = web_sys::window().unwrap();
                        let width = win.inner_width().unwrap().as_f64().unwrap() as u32;
                        let height = win.inner_height().unwrap().as_f64().unwrap() as u32;
                        size = PhysicalSize::new(width, height);
                        state.scale_factor = state.window.scale_factor();

                        state.config.width = width;
                        state.config.height = height;
                        // state.config.width = (width as f64 * state.scale_factor) as u32;
                        // state.config.height = (height as f64 * state.scale_factor) as u32;
                        state.surface.configure(&state.device, &state.config);
                    }
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        state.scale_factor = *scale_factor;
                    }
                    WindowEvent::RedrawRequested => {
                        state.window.request_redraw();

                        if !surface_configured {
                            return;
                        }

                        state.queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&fingers_to_buffer(&fingers)));
                        state.queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[
                            CameraInstance {
                                dimensions: [size.width as f32, size.height as f32],
                                scale_factor: [state.scale_factor as f32, state.scale_factor as f32],
                            }
                        ]));

                        let output = state.surface.get_current_texture().unwrap();
                        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Render Encoder"),
                        });
                        {
                            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("Render Pass"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(Color {
                                            r: 0.0,
                                            g: 0.2,
                                            b: 0.4,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    }
                                })],
                                depth_stencil_attachment: None,
                                occlusion_query_set: None,
                                timestamp_writes: None,
                            });

                            render_pass.set_pipeline(&state.render_pipeline);
                            render_pass.set_bind_group(0, &camera_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
                            render_pass.draw(0..6, 0..MAX_FINGERS);
                        }
                        state.queue.submit(std::iter::once(encoder.finish()));

                        output.present();
                    }
                    WindowEvent::Touch(touch) => {
                        match touch.phase {
                            TouchPhase::Started |
                            TouchPhase::Moved => {
                                fingers.insert(touch.id, *touch);
                                // send_fingers(&fingers);
                            }
                            TouchPhase::Ended |
                            TouchPhase::Cancelled => {
                                fingers.remove(&touch.id);
                            }
                        }
                        info!("{:?}", fingers);
                    }
                    _ => {}
                }
            },
            _ => {}
        }
    }).unwrap();
}

fn physical_to_vector(position: &PhysicalPosition<f64>) -> Vector2<f64> {
    Vector2::new(position.x, position.y)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct TouchInstance {
    position: [f32; 2],
    force: f32,
    scale: f32,
}

impl TouchInstance {
    fn empty() -> Self {
        Self {
            position: [0.0, 0.0],
            force: 0.0,
            scale: 0.0,
        }
    }
}

impl TouchInstance {
    pub fn from_touch(touch: &Touch) -> Self {
        let position = [touch.location.x as f32, touch.location.y as f32];
        let force = touch.force.unwrap_or(Force::Normalized(0.0)).normalized() as f32;
        let scale = 100.0;
        Self {
            position,
            force,
            scale,
        }
    }

    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<TouchInstance>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 2]>() as BufferAddress,
                    shader_location: 1,
                    format: Float32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 2,
                    format: Float32,
                },
            ],
        }
    }
}

fn fingers_to_buffer(fingers: &Fingers) -> Vec<TouchInstance> {
    let mut result = vec![];
    result.resize_with(10, || {
        TouchInstance::empty()
    });

    let mut index = 0;
    for (_id, finger) in &fingers.fingers {
        result[index] = TouchInstance::from_touch(&finger.0);
        index += 1;
    }

    result
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
struct CameraInstance {
    dimensions: [f32; 2],
    scale_factor: [f32; 2],
}

fn send_fingers(fingers: &Fingers) {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let url = format!("/log?q={}", serde_json::to_string_pretty(fingers).unwrap_or("couldn't serialize".into()));

    if let Ok(request) = Request::new_with_str_and_init(&url, &opts) {
        if request
            .headers()
            .set("Accept", "application/vnd.github.v3+json")
            .is_err() {
            info!("uh oh");
            return;
        }
        let window = web_sys::window().unwrap();
        let _ = window.fetch_with_request(&request);
    }
}