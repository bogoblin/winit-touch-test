use std::collections::HashMap;
use std::future::Future;
use std::mem;
use std::sync::Arc;
use log::info;
use wasm_bindgen::prelude::wasm_bindgen;
use wgpu::VertexFormat::{Float32, Float32x2};
use wgpu::{include_wgsl, BindGroup, Buffer, BufferAddress, Color, CompositeAlphaMode, PresentMode, VertexBufferLayout};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{FingerId, PointerKind, PointerSource, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scale_factor: f64,
    render_pipeline: wgpu::RenderPipeline,
    surface_configured: bool,
    size: PhysicalSize<u32>,
    fingers: Fingers,
    instance_buffer: Buffer,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    window: Arc<Box<dyn Window>>,
}

#[derive(Debug)]
struct Fingers {
    fingers: HashMap<FingerId, PhysicalPosition<f64>>
}

impl Fingers {
    pub(crate) fn remove(&mut self, id: &FingerId) {
        self.fingers.remove(id);
    }
    pub(crate) fn insert(&mut self, id: FingerId, position: PhysicalPosition<f64>) {
        self.fingers.insert(id, position);
    }
    fn new() -> Self {
        Self {
            fingers: Default::default(),
        }
    }
}

struct App {
    window: Option<Arc<Box<dyn Window>>>,
    state: Option<State>,
    receiver: Option<futures::channel::oneshot::Receiver<State>>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            state: None,
            receiver: None,
        }
    }
}

impl ApplicationHandler for App {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if let Ok(window) = event_loop.create_window(
            WindowAttributes::default()
                .with_surface_size(PhysicalSize::new(1280, 720))
        ) {
            let window_handle = Arc::new(window);
            self.window = Some(window_handle.clone());
            let (sender, receiver) = futures::channel::oneshot::channel();
            self.receiver = Some(receiver);
            wasm_bindgen_futures::spawn_local(async move {
                let state = State::new(window_handle.clone()).await;
                let _ = sender.send(state);
            });
        }
    }

    fn window_event(&mut self, _event_loop: &dyn ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        if self.state.is_none() {
            if let Some(receiver) = &mut self.receiver {
                if let Ok(Some(mut state)) = receiver.try_recv() {
                    state.resize();
                    info!("resized");
                    self.state = Some(state);
                }
            }
        }
        if let Some(state) = &mut self.state {
            state.handle_window_event(event);
        }
    }
}

impl State {
    const MAX_FINGERS: u32 = 10;

    pub fn new(window: Arc<Box<dyn Window>>) -> impl Future<Output = Self> + 'static {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();
        async move {
            let adapter = instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                },
            ).await.unwrap();

            let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults();
            required_limits.max_texture_dimension_2d = 4096;
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
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: window.surface_size().width,
                height: window.surface_size().height,
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

            let instance_buffer = device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: None,
                    size: (State::MAX_FINGERS as usize * mem::size_of::<TouchInstance>()) as BufferAddress,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );

            #[cfg(target_arch = "wasm32")]
            {
                use winit::platform::web::WindowExtWeb;
                web_sys::window()
                    .and_then(|win| win.document())
                    .and_then(|doc| {
                        let dst = doc.get_element_by_id("wasm-example").unwrap();
                        let canvas = web_sys::Element::from(window.canvas().unwrap().clone());
                        dst.append_child(&canvas).ok().unwrap();
                        Some(())
                    })
                    .expect("Couldn't append canvas to document body.");
            }

            State {
                surface,
                device,
                queue,
                config,
                scale_factor: 1.0,
                render_pipeline,
                surface_configured: false,
                size: Default::default(),
                fingers: Fingers::new(),
                instance_buffer,
                camera_buffer,
                camera_bind_group,
                window,
            }
        }
    }

    fn resize(&mut self) {
        let win = web_sys::window().unwrap();
        let width = win.inner_width().unwrap().as_f64().unwrap() as u32;
        let height = win.inner_height().unwrap().as_f64().unwrap() as u32;
        self.size = PhysicalSize::new(width, height);
        self.scale_factor = self.window.scale_factor();
        self.config.width = (width as f64 * self.scale_factor) as u32;
        self.config.height = (height as f64 * self.scale_factor) as u32;
        info!("{} x {} at {}", width, height, self.scale_factor);
        self.surface.configure(&self.device, &self.config);
        self.surface_configured = true;
    }

    fn handle_window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::SurfaceResized(..) => {
                self.resize();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
            }
            WindowEvent::RedrawRequested => {
                self.window.request_redraw();
                
                if !self.surface_configured {
                    return;
                }

                self.queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&fingers_to_buffer(&self.fingers)));
                self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[
                    CameraInstance {
                        dimensions: [self.size.width as f32, self.size.height as f32],
                        scale_factor: [self.scale_factor as f32, self.scale_factor as f32],
                    }
                ]));

                let output = self.surface.get_current_texture().unwrap();
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

                    render_pass.set_pipeline(&self.render_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
                    render_pass.draw(0..6, 0..Self::MAX_FINGERS);
                }
                self.queue.submit(std::iter::once(encoder.finish()));

                output.present();
            }
            WindowEvent::PointerMoved { position, source, .. } => {
                match source {
                    PointerSource::Mouse => {}
                    PointerSource::Touch { finger_id, .. } => {
                        self.fingers.insert(finger_id, position);
                        info!("{:?}", self.fingers);
                    }
                    PointerSource::Unknown => {}
                }
            }
            WindowEvent::PointerLeft { kind, .. } => {
                match kind {
                    PointerKind::Mouse => {}
                    PointerKind::Touch(finger_id) => {
                        self.fingers.remove(&finger_id);
                    }
                    PointerKind::Unknown => {}
                }
            }
            _ => {}
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().unwrap();
    info!("asdfasdfasdf");
    let event_loop = EventLoop::new().unwrap();

    use winit::platform::web::EventLoopExtWeb;
    let app = App::new();
    event_loop.spawn_app(app);
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
    pub fn from_touch(location: PhysicalPosition<f64>) -> Self {
        let position = [location.x as f32, location.y as f32];
        let force = 0.0; // todo: maybe we need force later
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
        result[index] = TouchInstance::from_touch(finger.clone());
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