use ash::{VkResult, vk::{self, Flags}, vk_bitflags_wrapped};
use std::ffi::CStr;
use std::path::Path;

#[cfg(target_vendor = "apple")]
mod metalfx;

#[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
mod dlss;

#[derive(Clone, Copy)]
pub struct SuperResolutionApplicationInfo<'a> {
    /// A stable project/application identifier registered with the vendor. For
    /// NGX this is the DLSS Project ID (a GUID string).
    pub project_id: &'a CStr,
    /// The application/engine version string.
    pub engine_version: &'a CStr,
    /// A writable directory the backend may use for its own data (caches, logs).
    /// For NGX this is the `ApplicationDataPath`. Must already exist and be
    /// writable.
    pub application_data_path: &'a Path,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SuperResolutionEngine(u32);
impl SuperResolutionEngine {
    /// The single engine exposed by the DLSS backend: DLSS-RR (Ray Reconstruction),
    /// a temporal denoising upscaler.
    pub const DLSS_RR: SuperResolutionEngine = SuperResolutionEngine(0);

    /// `MTL4FXTemporalScaler` — temporal upscaling that accumulates samples across
    /// frames using motion vectors. Corresponds to a temporal engine.
    pub const METALFX_TEMPORAL_SCALER: SuperResolutionEngine = SuperResolutionEngine(10);

    /// `MTL4FXSpatialScaler` — single-frame spatial upscaling with no history.
    pub const METALFX_SPATIAL_SCALER: SuperResolutionEngine = SuperResolutionEngine(11);

    /// `MTL4FXTemporalDenoisedScaler` — temporal upscaling combined with denoising,
    /// intended for ray-traced inputs.
    pub const METALFX_TEMPORAL_DENOISED_SCALER: SuperResolutionEngine = SuperResolutionEngine(12);
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperResolutionEnginePropertyFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SuperResolutionEnginePropertyFlags, Flags);
impl SuperResolutionEnginePropertyFlags {
    /// The implementation accumulates samples across multiple frames.
    pub const IS_TEMPORAL: Self = Self(0x0000_0001);
    /// A pipeline cache may be passed to `create_super_resolution_session` to
    /// improve session creation performance.
    pub const SUPPORTS_PIPELINE_CACHE: Self = Self(0x0000_0002);
    /// The engine can use any scaling factor up to the maximum supported one,
    /// with source and destination region sizes set independently.
    pub const SUPPORTS_CONTINUOUS_SCALING: Self = Self(0x0000_0004);
    /// The dispatch `source_size` may differ from the session's
    /// `max_source_region_size`.
    pub const SUPPORTS_DYNAMIC_SOURCE_SIZE: Self = Self(0x0000_0008);
    /// The engine uses the `sharpness` value provided at dispatch.
    pub const USES_SHARPNESS: Self = Self(0x0000_0010);
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperResolutionImageUseFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SuperResolutionImageUseFlags, Flags);
impl SuperResolutionImageUseFlags {
    /// Used as a source color image during a dispatch.
    pub const SOURCE: Self = Self(0x0000_0001);
    /// Used as a destination color image during a dispatch.
    pub const DESTINATION: Self = Self(0x0000_0002);
    /// Used as a depth image during a dispatch.
    pub const DEPTH: Self = Self(0x0000_0004);
    /// Used as a motion vector image during a dispatch.
    pub const MOTION_VECTORS: Self = Self(0x0000_0008);
    /// Used as a reactive mask image during a dispatch.
    pub const REACTIVE_MASK: Self = Self(0x0000_0010);
    /// Used as an ignore history mask image during a dispatch.
    pub const IGNORE_HISTORY_MASK: Self = Self(0x0000_0020);
    /// Used as an exposure scale image during a dispatch.
    pub const EXPOSURE_SCALE: Self = Self(0x0000_0040);
    /// Used as a diffuse albedo G-buffer image by a denoising engine.
    pub const DIFFUSE_ALBEDO: Self = Self(0x0000_0080);
    /// Used as a specular albedo G-buffer image by a denoising engine.
    pub const SPECULAR_ALBEDO: Self = Self(0x0000_0100);
    /// Used as a world-space normal G-buffer image by a denoising engine.
    pub const NORMAL: Self = Self(0x0000_0200);
    /// Used as a roughness G-buffer image by a denoising engine.
    pub const ROUGHNESS: Self = Self(0x0000_0400);
    /// Used as a specular hit-distance image by a denoising engine.
    pub const SPECULAR_HIT_DISTANCE: Self = Self(0x0000_0800);
    /// Used as a denoise strength mask image by a denoising engine.
    pub const DENOISE_STRENGTH_MASK: Self = Self(0x0000_1000);
    /// Used as a transparency overlay image by a denoising engine.
    pub const TRANSPARENCY_OVERLAY: Self = Self(0x0000_2000);
}


#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperResolutionQualityFocusFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SuperResolutionQualityFocusFlags, Flags);
impl SuperResolutionQualityFocusFlags {
    /// Strives for a balance of quality, performance, and power consumption.
    pub const BALANCED: Self = Self(0x0000_0001);
    /// Focuses on destination image quality, potentially at the expense of
    /// performance and power consumption.
    pub const QUALITY: Self = Self(0x0000_0002);
    /// Focuses on speed of execution, potentially at the expense of power
    /// consumption and image quality.
    pub const PERFORMANCE: Self = Self(0x0000_0004);
    /// Focuses on reduced power consumption, potentially at the expense of
    /// performance and image quality.
    pub const POWER: Self = Self(0x0000_0008);
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperResolutionSessionCreateFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SuperResolutionSessionCreateFlags, Flags);
impl SuperResolutionSessionCreateFlags {
    /// The dispatch `source_size` may vary between subsequent dispatches.
    pub const DYNAMIC_SOURCE_SIZE: Self = Self(0x0000_0001);
    /// The motion vectors image is the size of the destination image rather
    /// than the source image.
    pub const MOTION_VECTORS_USE_DESTINATION_DIMENSIONS: Self = Self(0x0000_0002);
    /// The motion vectors image is rendered with the camera jitter offset.
    pub const MOTION_VECTORS_USE_JITTER: Self = Self(0x0000_0004);
    /// Scaling may target a subregion of the destination image, rather than
    /// requiring the destination region to match the full destination image.
    pub const ALLOW_SUBRECT_DESTINATION: Self = Self(0x0000_0008);
    /// The supplied depth buffer contains linear depth values.
    pub const LINEAR_DEPTH: Self = Self(0x0000_0010);
    /// Inverted depth tracking: high depth values are near the camera.
    pub const INVERTED_DEPTH: Self = Self(0x0000_0020);
    /// The `sharpness` value is honored; otherwise it is ignored.
    pub const USE_SHARPNESS: Self = Self(0x0000_0040);
    /// Apply auto exposure, ignoring application-provided exposure values.
    pub const USE_AUTO_EXPOSURE: Self = Self(0x0000_0080);
    /// Source and output images contain only LDR color data (0.0 to 1.0).
    pub const FORCE_LDR_COLORS: Self = Self(0x0000_0100);
    /// The application is using descriptor sets rather than descriptor heaps.
    pub const USE_LEGACY_DESCRIPTORS: Self = Self(0x0000_0200);
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperResolutionSessionMemoryRequirementsFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SuperResolutionSessionMemoryRequirementsFlags, Flags);
impl SuperResolutionSessionMemoryRequirementsFlags {
    /// The memory may persist across multiple dispatches on the session.
    pub const PERSISTENT: Self = Self(0x0000_0001);
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SuperResolutionDispatchFlags(pub(crate) Flags);
vk_bitflags_wrapped!(SuperResolutionDispatchFlags, Flags);
impl SuperResolutionDispatchFlags {
    /// The camera projection maps the near plane to depth `1.0` and the far
    /// plane to `0.0`, instead of the default `0.0`/`1.0` mapping.
    pub const INVERTED_DEPTH_RANGE: Self = Self(0x0000_0001);
    /// The camera's view-projection matrix uses orthographic projection rather
    /// than perspective projection.
    pub const ORTHOGRAPHIC_PROJECTION: Self = Self(0x0000_0002);
    /// Motion history is ignored for this dispatch, e.g. across a cut-scene
    /// transition.
    pub const RESET_HISTORY: Self = Self(0x0000_0004);
}

const MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT: usize = 8;
const MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT: usize = 16;
const MAX_SUPER_RESOLUTION_NAME_SIZE: usize = 32;

pub struct SuperResolutionEngineProperties {
    pub vendor_id: u32,
    pub engine_version: u32,
    pub engine_uuid: [u8; vk::UUID_SIZE],
    pub engine_name: [std::ffi::c_char; MAX_SUPER_RESOLUTION_NAME_SIZE],
    pub flags: SuperResolutionEnginePropertyFlags,
    pub image_used: SuperResolutionImageUseFlags,
    pub supported_quality_focuses: SuperResolutionQualityFocusFlags,
    pub supported_queue_family_indexes: [u32; MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT],
    pub supported_scaling_factor_count: u32,
    pub supported_scaling_factors: [ScalingFactor; MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT],
    pub max_destination_region_size: vk::Extent2D,
    pub max_supported_concurrent_session_dispatches: u32,
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalingFactor {
    pub numerator: u32,
    pub denominator: u32
}

/// Format and usage flags supported by a super resolution engine for a
/// particular image use.
///
/// The optional source/auxiliary formats use [`vk::Format::UNDEFINED`] to
/// indicate that the corresponding image will not be provided to dispatches.
pub struct SuperResolutionSessionCreateInfo {
    /// The engine for which the session will be created.
    pub engine: SuperResolutionEngine,
    /// Additional session parameters.
    pub flags: SuperResolutionSessionCreateFlags,
    /// The quality focuses that must be supported by the session. Enabling
    /// fewer bits may reduce session creation time and memory.
    pub required_quality_focuses: SuperResolutionQualityFocusFlags,
    /// Format of the destination color image.
    pub destination_format: vk::Format,
    /// Format of the source color image.
    pub source_format: vk::Format,
    /// Format of the source depth image, or [`vk::Format::UNDEFINED`] if none.
    pub source_depth_format: vk::Format,
    /// Format of the motion vectors image, or [`vk::Format::UNDEFINED`] if none.
    pub motion_vector_format: vk::Format,
    /// Format of the reactive mask image, or [`vk::Format::UNDEFINED`] if none.
    pub reactive_mask_format: vk::Format,
    /// Format of the ignore history mask image, or [`vk::Format::UNDEFINED`] if none.
    pub ignore_history_mask_format: vk::Format,
    /// Format of the 1x1 exposure scale image, or [`vk::Format::UNDEFINED`] if none.
    pub exposure_scale_format: vk::Format,
    /// Format of the diffuse albedo G-buffer image (denoising engines only), or
    /// [`vk::Format::UNDEFINED`] if none.
    pub diffuse_albedo_format: vk::Format,
    /// Format of the specular albedo G-buffer image (denoising engines only), or
    /// [`vk::Format::UNDEFINED`] if none.
    pub specular_albedo_format: vk::Format,
    /// Format of the world-space normal G-buffer image (denoising engines only),
    /// or [`vk::Format::UNDEFINED`] if none.
    pub normal_format: vk::Format,
    /// Format of the roughness G-buffer image (denoising engines only), or
    /// [`vk::Format::UNDEFINED`] if none.
    pub roughness_format: vk::Format,
    /// Format of the specular hit-distance image (denoising engines only), or
    /// [`vk::Format::UNDEFINED`] if none.
    pub specular_hit_distance_format: vk::Format,
    /// Format of the denoise strength mask image (denoising engines only), or
    /// [`vk::Format::UNDEFINED`] if none.
    pub denoise_strength_mask_format: vk::Format,
    /// Format of the transparency overlay image (denoising engines only), or
    /// [`vk::Format::UNDEFINED`] if none.
    pub transparency_overlay_format: vk::Format,
    /// Size of the destination region, which may be smaller than the
    /// destination image.
    pub destination_region_size: vk::Extent2D,
    /// Maximum size of the source content (color, depth, and optional masks).
    pub max_source_region_size: vk::Extent2D,
    /// Scale applied to the X axis of motion vectors to bring them into source
    /// image space. Use 1.0 if already in source space.
    pub motion_vector_scale_x: f32,
    /// Scale applied to the Y axis of motion vectors to bring them into source
    /// image space. Use 1.0 if already in source space.
    pub motion_vector_scale_y: f32,
    /// Maximum number of dispatches the application may execute concurrently.
    pub max_concurrent_dispatches: u32,
}

pub struct SuperResolutionImageProperties {
    pub format: vk::Format,
    pub image_usage_flags: vk::ImageUsageFlags,
}

/// Describes a block of internal device memory required by a super resolution
/// session.
pub struct SuperResolutionSessionMemoryRequirements {
    /// Additional characteristics of the required memory.
    pub flags: SuperResolutionSessionMemoryRequirementsFlags,
    /// The size, alignment, and acceptable memory types for this bind point.
    pub memory_requirements: vk::MemoryRequirements,
}

/// Describes device memory to attach to one of a session's bind points.
pub struct BindSuperResolutionSessionMemoryInfo {
    /// Index of the bind point to which `memory` is attached. Corresponds to
    /// the position of the matching entry returned by
    /// [`SuperResolutionDevice::get_super_resolution_session_memory_requirements`].
    pub object_index: u32,
    /// The device memory to attach.
    pub memory: vk::DeviceMemory,
    /// Start offset within `memory` of the region to bind.
    pub memory_offset: vk::DeviceSize,
}

/// Requirements for ranges within the resource and sampler descriptor heaps
/// that a super resolution session uses for its internal descriptors.
pub struct SuperResolutionDescriptorHeapRanges {
    /// Required size of a range on the resource descriptor heap.
    pub resource_heap_size: vk::DeviceSize,
    /// Required alignment of that resource heap range.
    pub resource_heap_alignment: vk::DeviceSize,
    /// Required size of a range on the sampler descriptor heap.
    pub sampler_heap_size: vk::DeviceSize,
    /// Required alignment of that sampler heap range.
    pub sampler_heap_alignment: vk::DeviceSize,
}


pub trait SuperResolutionPhysicalDevice {
    fn enumerate_super_resolution_engines(
        &self,
        application_info: &SuperResolutionApplicationInfo<'_>,
    ) -> VkResult<Vec<SuperResolutionEngine>>;

    fn get_super_resolution_engine_properties(
        &self,
        engine: SuperResolutionEngine,
    ) -> SuperResolutionEngineProperties;

    /// Enumerates the image properties supported by a super resolution `engine`
    /// for the given `image_use`.
    fn get_super_resolution_engine_supported_image_properties(
        &self,
        engine: SuperResolutionEngine,
        image_use: SuperResolutionImageUseFlags,
    ) -> VkResult<Vec<SuperResolutionImageProperties>>;

}

impl SuperResolutionPhysicalDevice for pumicite::physical_device::PhysicalDevice {
        fn enumerate_super_resolution_engines(
        &self,
        application_info: &SuperResolutionApplicationInfo<'_>,
    ) -> VkResult<Vec<SuperResolutionEngine>> {
        #[cfg(target_vendor = "apple")]
        {
            let _ = application_info;
            return Ok(metalfx::ENGINES.to_vec());
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return Ok(dlss::enumerate_engines(self, application_info));
        }
        #[allow(unreachable_code)]
        return Ok(Vec::new())
    }

    fn get_super_resolution_engine_properties(
        &self,
        engine: SuperResolutionEngine,
    ) -> SuperResolutionEngineProperties {
        #[cfg(target_vendor = "apple")]
        {
            return metalfx::engine_properties(self, engine);
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return dlss::engine_properties(self, engine);
        }
        panic!("Unknown upscaler engine");
    }

    /// Enumerates the image properties supported by a super resolution `engine`
    /// for the given `image_use`.
    fn get_super_resolution_engine_supported_image_properties(
        &self,
        engine: SuperResolutionEngine,
        image_use: SuperResolutionImageUseFlags,
    ) -> VkResult<Vec<SuperResolutionImageProperties>> {
        #[cfg(target_vendor = "apple")]
        {
            return Ok(metalfx::engine_supported_image_properties(engine, image_use));
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return Ok(dlss::engine_supported_image_properties(engine, image_use));
        }
        #[allow(unreachable_code)]
        Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
    }
}

pub trait SuperResolutionDevice {
    /// Queries the internal device memory requirements of a super resolution
    /// session that would be created with `session_create_info`.
    fn get_super_resolution_session_memory_requirements(
        &self,
        session_create_info: &SuperResolutionSessionCreateInfo,
    ) -> VkResult<Vec<SuperResolutionSessionMemoryRequirements>>;

    /// Queries the descriptor heap range requirements of a super resolution
    /// session that would be created with `session_create_info`.
    fn get_super_resolution_session_descriptor_heap_ranges(
        &self,
        session_create_info: &SuperResolutionSessionCreateInfo,
    ) -> VkResult<SuperResolutionDescriptorHeapRanges>;
}

impl SuperResolutionDevice for pumicite::Device {
    fn get_super_resolution_session_memory_requirements(
        &self,
        _session_create_info: &SuperResolutionSessionCreateInfo,
    ) -> VkResult<Vec<SuperResolutionSessionMemoryRequirements>> {
        #[cfg(target_vendor = "apple")]
        {
            return Ok(metalfx::session_memory_requirements());
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return Ok(dlss::session_memory_requirements());
        }
        #[allow(unreachable_code)]
        Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
    }

    fn get_super_resolution_session_descriptor_heap_ranges(
        &self,
        _session_create_info: &SuperResolutionSessionCreateInfo,
    ) -> VkResult<SuperResolutionDescriptorHeapRanges> {
        #[cfg(target_vendor = "apple")]
        {
            return Ok(metalfx::session_descriptor_heap_ranges());
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return Ok(dlss::session_descriptor_heap_ranges());
        }
        #[allow(unreachable_code)]
        Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
    }
}

/// An instance of a specific scaler implementation, created from a
/// [`SuperResolutionSessionCreateInfo`].
///
/// Wraps `VkSuperResolutionSessionEXT`. Owns the [`Device`](pumicite::Device)
/// it was created from. Before it can be dispatched to, the session must have
/// its memory requirements satisfied (see
/// [`SuperResolutionDevice::get_super_resolution_session_memory_requirements`])
/// and be initialized.
pub struct SuperResolutionSession {
    device: pumicite::Device,
    /// The backing MetalFX scaler. Retained for the session's lifetime.
    #[cfg(target_vendor = "apple")]
    scaler: metalfx::Scaler,
    /// The backing DLSS-RR session (deferred NGX feature + create params).
    #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
    dlss: dlss::DlssSession,
}

impl pumicite::HasDevice for SuperResolutionSession {
    fn device(&self) -> &pumicite::Device {
        &self.device
    }
}

impl SuperResolutionSession {
    /// Creates a super resolution session.
    pub fn new(
        pipeline_cache: &pumicite::pipeline::PipelineCache,
        create_info: &SuperResolutionSessionCreateInfo,
        application_info: &SuperResolutionApplicationInfo<'_>,
    ) -> VkResult<SuperResolutionSession> {
        #[cfg(target_vendor = "apple")]
        {
            let _ = application_info; // MetalFX has no application identity.
            return metalfx::create_session(pipeline_cache, create_info);
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return dlss::create_session(pipeline_cache, create_info, application_info);
        }
        // No backend owns this engine (none were enumerated).
        #[cfg(not(any(target_vendor = "apple", feature = "dlss")))]
        {
            let _ = application_info;
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
        }
    }

    /// Attaches device memory to this session's internal bind points.
    pub fn bind_memory(
        &self,
        _bind_infos: &[BindSuperResolutionSessionMemoryInfo],
    ) -> VkResult<()> {
        #[cfg(target_vendor = "apple")]
        {
            return Ok(());
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            // NGX manages all session memory internally; nothing to bind.
            return Ok(());
        }
    }

    /// Queries the recommended per-frame jitter pattern for this session.
    /// If the engine does not use jitter (i.e. it is not temporal), this
    /// returns `Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)`.
    pub fn recommended_jitter_pattern(
        &self,
        destination_region_size: vk::Extent2D,
        source_region_size: vk::Extent2D,
    ) -> VkResult<Vec<(f32, f32)>> {
        #[cfg(target_vendor = "apple")]
        {
            return metalfx::recommended_jitter_pattern(
                self,
                destination_region_size,
                source_region_size,
            );
        }
        #[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            return dlss::recommended_jitter_pattern(destination_region_size, source_region_size);
        }
        #[allow(unreachable_code)]
        Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
    }
}

/// Describes an image passed into a super resolution dispatch.
pub struct SuperResolutionImageInfo<'a> {
    /// The image content, specified as an image view create info.
    pub view: &'a vk::ImageViewCreateInfo<'a>,
    /// Offset, in texels, to the region within the image used by the dispatch.
    pub view_offset: vk::Offset2D,
    /// Layout of the image at the start of the dispatch.
    pub initial_layout: vk::ImageLayout,
    /// Layout the image is expected to be in once the dispatch completes.
    pub final_layout: vk::ImageLayout,
}

/// Camera parameters used to render the scene.
pub struct SuperResolutionCameraInfo {
    /// Column-major view-projection matrix (world space to clip space),
    /// excluding jitter.
    pub view_projection_matrix: [[f32; 4]; 4],
    /// Near plane of the camera frustum.
    pub near: f32,
    /// Far plane of the camera frustum.
    pub far: f32,
    /// Field of view of the camera frustum.
    pub fov: f32,
}

/// Motion information for a super resolution dispatch.
pub struct SuperResolutionDispatchMotionInfo<'a> {
    /// Image encoding the motion of pixels between frames, or `None` if
    /// unavailable or unused by the engine.
    pub motion_vectors_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Image encoding reactive information, or `None` if unavailable or unused.
    pub reactive_mask_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Image encoding whether per-pixel history should be considered, or `None`
    /// if unavailable or unused.
    pub ignore_history_mask_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Information about the camera used to render the scene.
    pub camera_info: SuperResolutionCameraInfo,
    /// Per-frame jitter applied in the X dimension.
    pub texel_jitter_x: f32,
    /// Per-frame jitter applied in the Y dimension.
    pub texel_jitter_y: f32,
}

/// Exposure information for a super resolution dispatch.
pub struct SuperResolutionDispatchExposureInfo<'a> {
    /// Pre-exposure value for HDR images; ignored if not using HDR or if auto
    /// exposure is enabled.
    pub pre_exposure: f32,
    /// Uniform exposure scale factor for HDR images; ignored if
    /// `exposure_scale_image_info` is provided.
    pub exposure_scale_uniform: f32,
    /// 1x1 image providing an exposure scale factor for HDR images, or `None`
    /// to use `exposure_scale_uniform`.
    pub exposure_scale_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
}

/// G-buffer inputs for a denoising super resolution dispatch.
///
/// Only used by denoising engines (e.g. the MetalFX temporal denoised scaler).
/// The diffuse/specular albedo, normal, and roughness images are required; the
/// remaining images are optional and may be `None`.
pub struct SuperResolutionDispatchDenoiseInfo<'a> {
    /// Diffuse albedo G-buffer image.
    pub diffuse_albedo_image_info: &'a SuperResolutionImageInfo<'a>,
    /// Specular albedo G-buffer image.
    pub specular_albedo_image_info: &'a SuperResolutionImageInfo<'a>,
    /// World-space normal G-buffer image.
    pub normal_image_info: &'a SuperResolutionImageInfo<'a>,
    /// Roughness G-buffer image.
    pub roughness_image_info: &'a SuperResolutionImageInfo<'a>,
    /// Specular hit-distance image, or `None` if unused.
    pub specular_hit_distance_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Denoise strength mask image, or `None` if unused.
    pub denoise_strength_mask_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Transparency overlay image, or `None` if unused.
    pub transparency_overlay_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Column-major world-to-view (camera) matrix.
    pub world_to_view_matrix: [[f32; 4]; 4],
    /// Column-major view-to-clip (projection) matrix.
    pub view_to_clip_matrix: [[f32; 4]; 4],
}

/// Parameters driving a single super resolution upscaling dispatch.
pub struct SuperResolutionDispatchInfo<'a> {
    /// Index of the dispatch. Must be less than the session's
    /// `max_concurrent_dispatches`, and unique across overlapping concurrent
    /// dispatches (including across swapchain images).
    pub dispatch_index: u32,
    /// Additional dispatch parameters.
    pub flags: SuperResolutionDispatchFlags,
    /// The single quality focus to use for this dispatch.
    pub quality_focus: SuperResolutionQualityFocusFlags,
    /// The destination color image.
    pub destination_image_info: &'a SuperResolutionImageInfo<'a>,
    /// The source color image.
    pub source_image_info: &'a SuperResolutionImageInfo<'a>,
    /// The source depth image, or `None` if the engine does not use depth.
    pub source_depth_image_info: Option<&'a SuperResolutionImageInfo<'a>>,
    /// Size of the source content.
    pub source_size: vk::Extent2D,
    /// Amount of sharpening to apply, in the range 0.0 to 1.0.
    pub sharpness: f32,
    /// Motion information, or `None` if the engine is not temporal.
    pub motion_info: Option<&'a SuperResolutionDispatchMotionInfo<'a>>,
    /// Exposure information, or `None` to use auto/default exposure.
    pub exposure_info: Option<&'a SuperResolutionDispatchExposureInfo<'a>>,
    /// G-buffer information, or `None` if the engine does not denoise.
    pub denoise_info: Option<&'a SuperResolutionDispatchDenoiseInfo<'a>>,
    /// Offset into the bound resource descriptor heap reserved for the
    /// session's internal descriptors.
    pub resource_descriptor_heap_offset: vk::DeviceSize,
    /// Offset into the bound sampler descriptor heap reserved for the session's
    /// internal descriptors.
    pub sampler_descriptor_heap_offset: vk::DeviceSize,
}

pub trait SuperResolutionCommandEncoder {
    /// Records initialization of a super resolution `session` into the command
    /// buffer.
    fn initialize_super_resolution_session(&mut self, session: &SuperResolutionSession);

    /// Records a super resolution upscaling dispatch into the command buffer.
    fn dispatch_super_resolution(
        &mut self,
        session: &SuperResolutionSession,
        dispatch_info: &SuperResolutionDispatchInfo,
    );
}

impl SuperResolutionCommandEncoder for pumicite::command::CommandEncoder<'_> {
    fn initialize_super_resolution_session(&mut self, session: &SuperResolutionSession) {
        #[cfg(target_vendor = "apple")]
        {
            metalfx::initialize_session(self, session);
            return;
        }
#[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            dlss::initialize_session(self, session);
            return;
        }
    }

    fn dispatch_super_resolution(
        &mut self,
        session: &SuperResolutionSession,
        dispatch_info: &SuperResolutionDispatchInfo,
    ) {
        #[cfg(target_vendor = "apple")]
        {
            metalfx::dispatch(self, session, dispatch_info);
            return;
        }
#[cfg(all(not(target_vendor = "apple"), feature = "dlss"))]
        {
            dlss::dispatch(self, session, dispatch_info);
            return;
        }
    }
}
