//! MetalFX-backed implementation of the super-resolution interface.
//!
//! Apple exposes three distinct upscaling effects, each surfaced here as a
//! unique [`SuperResolutionEngine`]:
//!
//! | Engine const                  | Metal class                     |
//! |-------------------------------|---------------------------------|
//! | [`TEMPORAL_SCALER`]           | `MTL4FXTemporalScaler`          |
//! | [`SPATIAL_SCALER`]            | `MTL4FXSpatialScaler`           |
//! | [`TEMPORAL_DENOISED_SCALER`]  | `MTL4FXTemporalDenoisedScaler`  |

use core::ffi::c_char;

use ash::{VkResult, vk};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CompilerDescriptor, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLTexture,
};
use objc2_metal_fx::{
    MTL4FXSpatialScaler, MTL4FXTemporalDenoisedScaler, MTL4FXTemporalScaler,
    MTLFXSpatialScalerBase, MTLFXSpatialScalerColorProcessingMode, MTLFXSpatialScalerDescriptor,
    MTLFXTemporalDenoisedScalerBase, MTLFXTemporalDenoisedScalerDescriptor, MTLFXTemporalScalerBase,
    MTLFXTemporalScalerDescriptor,
};
use pumicite::HasDevice;
use pumicite::command::CommandEncoder;
use pumicite::physical_device::PhysicalDevice;
use pumicite::pipeline::PipelineCache;
use pumicite::utils::AsMTLCommandBuffer;

use crate::{
    MAX_SUPER_RESOLUTION_NAME_SIZE, MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT,
    MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT, ScalingFactor, SuperResolutionDescriptorHeapRanges,
    SuperResolutionDispatchFlags, SuperResolutionDispatchInfo, SuperResolutionEngine,
    SuperResolutionEngineProperties, SuperResolutionEnginePropertyFlags,
    SuperResolutionImageInfo, SuperResolutionImageProperties, SuperResolutionImageUseFlags,
    SuperResolutionQualityFocusFlags, SuperResolutionSessionCreateFlags,
    SuperResolutionSessionCreateInfo, SuperResolutionSessionMemoryRequirements,
};

/// Returns the system default Metal device that MetalFX targets.
///
/// `MTLCreateSystemDefaultDevice` always returns the same device, so the backend
/// never needs to bridge an `MTLDevice` out of a Vulkan physical/logical device.
fn system_default_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
    MTLCreateSystemDefaultDevice().expect("no system default Metal device available")
}

/// `MTL4FXTemporalScaler` — temporal upscaling that accumulates samples across
/// frames using motion vectors. Corresponds to a temporal engine.
pub const TEMPORAL_SCALER: SuperResolutionEngine = SuperResolutionEngine(0);

/// `MTL4FXSpatialScaler` — single-frame spatial upscaling with no history.
pub const SPATIAL_SCALER: SuperResolutionEngine = SuperResolutionEngine(1);

/// `MTL4FXTemporalDenoisedScaler` — temporal upscaling combined with denoising,
/// intended for ray-traced inputs.
pub const TEMPORAL_DENOISED_SCALER: SuperResolutionEngine = SuperResolutionEngine(2);

/// The engines provided by the MetalFX backend, in enumeration order.
pub(crate) const ENGINES: [SuperResolutionEngine; 3] =
    [TEMPORAL_SCALER, SPATIAL_SCALER, TEMPORAL_DENOISED_SCALER];

/// Representative discrete scaling factors we expose for engines whose native
/// scaling is continuous. Each entry is an output-over-input ratio; the set is
/// filtered against the engine's reported `[min, max]` range at query time.
const CANDIDATE_SCALING_FACTORS: [ScalingFactor; 6] = [
    ScalingFactor { numerator: 3, denominator: 2 }, // 1.50x
    ScalingFactor { numerator: 5, denominator: 3 }, // 1.67x
    ScalingFactor { numerator: 7, denominator: 4 }, // 1.75x
    ScalingFactor { numerator: 2, denominator: 1 }, // 2.00x
    ScalingFactor { numerator: 5, denominator: 2 }, // 2.50x
    ScalingFactor { numerator: 3, denominator: 1 }, // 3.00x
];

/// Builds a fixed-size, NUL-padded engine name array from a byte string.
fn engine_name(name: &[u8]) -> [c_char; MAX_SUPER_RESOLUTION_NAME_SIZE] {
    debug_assert!(
        name.len() < MAX_SUPER_RESOLUTION_NAME_SIZE,
        "engine name must leave room for a NUL terminator",
    );
    let mut out = [0 as c_char; MAX_SUPER_RESOLUTION_NAME_SIZE];
    for (dst, &src) in out.iter_mut().zip(name) {
        *dst = src as c_char;
    }
    out
}

/// Selects the [`CANDIDATE_SCALING_FACTORS`] that fall within `[min, max]`,
/// writing them into `out` and returning how many were written.
fn discretize_scale_range(
    min: f32,
    max: f32,
    out: &mut [ScalingFactor; MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT],
) -> u32 {
    let mut count = 0usize;
    for factor in CANDIDATE_SCALING_FACTORS {
        let ratio = factor.numerator as f32 / factor.denominator as f32;
        if ratio >= min && ratio <= max && count < out.len() {
            out[count] = factor;
            count += 1;
        }
    }
    count as u32
}

/// Dispatches [`crate::SuperResolutionPhysicalDevice::get_super_resolution_engine_properties`]
/// to the per-engine MetalFX query.
pub(crate) fn engine_properties(
    physical_device: &PhysicalDevice,
    engine: SuperResolutionEngine,
) -> SuperResolutionEngineProperties {
    if engine == TEMPORAL_SCALER {
        temporal_scaler_properties(physical_device)
    } else if engine == SPATIAL_SCALER {
        spatial_scaler_properties(physical_device)
    } else if engine == TEMPORAL_DENOISED_SCALER {
        temporal_denoised_scaler_properties(physical_device)
    } else {
        panic!("unknown super resolution engine for the MetalFX backend")
    }
}

fn temporal_scaler_properties(
    physical_device: &PhysicalDevice,
) -> SuperResolutionEngineProperties {
    let device = system_default_device();

    // MetalFX temporal scaling is continuous; query the supported float range
    // and expose representative discrete factors within it.
    //
    // SAFETY: `device` is a valid `MTLDevice`, and these class methods have no
    // further preconditions.
    let (min_scale, max_scale) = unsafe {
        (
            MTLFXTemporalScalerDescriptor::supportedInputContentMinScaleForDevice(&device),
            MTLFXTemporalScalerDescriptor::supportedInputContentMaxScaleForDevice(&device),
        )
    };

    let mut supported_scaling_factors =
        [ScalingFactor { numerator: 0, denominator: 1 };
            MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT];
    let supported_scaling_factor_count =
        discretize_scale_range(min_scale, max_scale, &mut supported_scaling_factors);

    SuperResolutionEngineProperties {
        vendor_id: physical_device.properties().vendor_id,
        // MetalFX exposes no version/UUID; left as zero until a meaningful
        // source is identified.
        engine_version: 0,
        engine_uuid: [0u8; vk::UUID_SIZE],
        engine_name: engine_name(b"MetalFX Temporal Scaler"),
        flags: SuperResolutionEnginePropertyFlags::IS_TEMPORAL
            | SuperResolutionEnginePropertyFlags::SUPPORTS_CONTINUOUS_SCALING
            | SuperResolutionEnginePropertyFlags::SUPPORTS_DYNAMIC_SOURCE_SIZE
            | SuperResolutionEnginePropertyFlags::SUPPORTS_PIPELINE_CACHE,
        image_used: SuperResolutionImageUseFlags::SOURCE
            | SuperResolutionImageUseFlags::DESTINATION
            | SuperResolutionImageUseFlags::DEPTH
            | SuperResolutionImageUseFlags::MOTION_VECTORS
            | SuperResolutionImageUseFlags::REACTIVE_MASK
            | SuperResolutionImageUseFlags::EXPOSURE_SCALE,
        // MetalFX has no explicit quality presets; map to the single balanced
        // focus.
        supported_quality_focuses: SuperResolutionQualityFocusFlags::BALANCED,
        // Queue-family affinity is not yet determined; fill with the "ignored"
        // sentinel.
        supported_queue_family_indexes: [vk::QUEUE_FAMILY_IGNORED;
            MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT],
        supported_scaling_factor_count,
        supported_scaling_factors,
        // Metal's maximum 2D texture dimension on Apple silicon.
        max_destination_region_size: vk::Extent2D { width: 16384, height: 16384 },
        max_supported_concurrent_session_dispatches: 1,
    }
}

/// A texture input/output role of a temporal scaler, mapped from a
/// [`SuperResolutionImageUseFlags`] bit.
#[derive(Clone, Copy)]
enum TemporalTextureRole {
    Color,
    Output,
    Depth,
    Motion,
    ReactiveMask,
}

/// Color formats MetalFX accepts for both color inputs and the upscaled output.
const TEMPORAL_COLOR_FORMATS: &[vk::Format] = &[
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::B8G8R8A8_UNORM,
    vk::Format::A2B10G10R10_UNORM_PACK32,
];

impl TemporalTextureRole {
    /// The statically known set of supported formats for this role together
    /// with the Vulkan image usage MetalFX requires of that texture.
    ///
    /// MetalFX's accepted formats and minimal texture usage are fixed across
    /// every device that supports the temporal scaler, so this needs no device
    /// query or scaler instantiation. Inputs are sampled (`ShaderRead`); the
    /// output is written (`ShaderWrite`).
    fn supported(self) -> (&'static [vk::Format], vk::ImageUsageFlags) {
        match self {
            TemporalTextureRole::Color => {
                (TEMPORAL_COLOR_FORMATS, vk::ImageUsageFlags::SAMPLED)
            }
            TemporalTextureRole::Output => {
                (TEMPORAL_COLOR_FORMATS, vk::ImageUsageFlags::STORAGE)
            }
            TemporalTextureRole::Depth => (
                &[vk::Format::R32_SFLOAT, vk::Format::D32_SFLOAT],
                vk::ImageUsageFlags::SAMPLED,
            ),
            TemporalTextureRole::Motion => {
                (&[vk::Format::R16G16_SFLOAT], vk::ImageUsageFlags::SAMPLED)
            }
            TemporalTextureRole::ReactiveMask => {
                (&[vk::Format::R8_UNORM], vk::ImageUsageFlags::SAMPLED)
            }
        }
    }
}

fn temporal_denoised_scaler_properties(
    physical_device: &PhysicalDevice,
) -> SuperResolutionEngineProperties {
    let device = system_default_device();

    // SAFETY: `device` is a valid `MTLDevice`; these class methods have no
    // further preconditions.
    let (min_scale, max_scale) = unsafe {
        (
            MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(&device),
            MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(&device),
        )
    };

    let mut supported_scaling_factors =
        [ScalingFactor { numerator: 0, denominator: 1 };
            MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT];
    let supported_scaling_factor_count =
        discretize_scale_range(min_scale, max_scale, &mut supported_scaling_factors);

    SuperResolutionEngineProperties {
        vendor_id: physical_device.properties().vendor_id,
        engine_version: 0,
        engine_uuid: [0u8; vk::UUID_SIZE],
        engine_name: engine_name(b"MetalFX Denoised Scaler"),
        // The denoised scaler has no dynamic-resolution mode (its descriptor has
        // no input-content-properties toggle), so SUPPORTS_DYNAMIC_SOURCE_SIZE
        // is omitted.
        flags: SuperResolutionEnginePropertyFlags::IS_TEMPORAL
            | SuperResolutionEnginePropertyFlags::SUPPORTS_CONTINUOUS_SCALING
            | SuperResolutionEnginePropertyFlags::SUPPORTS_PIPELINE_CACHE,
        image_used: SuperResolutionImageUseFlags::SOURCE
            | SuperResolutionImageUseFlags::DESTINATION
            | SuperResolutionImageUseFlags::DEPTH
            | SuperResolutionImageUseFlags::MOTION_VECTORS
            | SuperResolutionImageUseFlags::REACTIVE_MASK
            | SuperResolutionImageUseFlags::EXPOSURE_SCALE
            | SuperResolutionImageUseFlags::DIFFUSE_ALBEDO
            | SuperResolutionImageUseFlags::SPECULAR_ALBEDO
            | SuperResolutionImageUseFlags::NORMAL
            | SuperResolutionImageUseFlags::ROUGHNESS
            | SuperResolutionImageUseFlags::SPECULAR_HIT_DISTANCE
            | SuperResolutionImageUseFlags::DENOISE_STRENGTH_MASK
            | SuperResolutionImageUseFlags::TRANSPARENCY_OVERLAY,
        supported_quality_focuses: SuperResolutionQualityFocusFlags::BALANCED,
        supported_queue_family_indexes: [vk::QUEUE_FAMILY_IGNORED;
            MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT],
        supported_scaling_factor_count,
        supported_scaling_factors,
        max_destination_region_size: vk::Extent2D { width: 16384, height: 16384 },
        max_supported_concurrent_session_dispatches: 1,
    }
}

fn spatial_scaler_properties(physical_device: &PhysicalDevice) -> SuperResolutionEngineProperties {
    // The spatial scaler exposes no scale-range query; it accepts arbitrary
    // input/output sizes, so we report the full candidate set and flag it
    // continuous.
    let mut supported_scaling_factors =
        [ScalingFactor { numerator: 0, denominator: 1 };
            MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT];
    let supported_scaling_factor_count =
        discretize_scale_range(1.0, f32::INFINITY, &mut supported_scaling_factors);

    SuperResolutionEngineProperties {
        vendor_id: physical_device.properties().vendor_id,
        engine_version: 0,
        engine_uuid: [0u8; vk::UUID_SIZE],
        engine_name: engine_name(b"MetalFX Spatial Scaler"),
        // Single-frame: not temporal. The per-dispatch input-content region makes
        // the source size dynamic.
        flags: SuperResolutionEnginePropertyFlags::SUPPORTS_CONTINUOUS_SCALING
            | SuperResolutionEnginePropertyFlags::SUPPORTS_DYNAMIC_SOURCE_SIZE
            | SuperResolutionEnginePropertyFlags::SUPPORTS_PIPELINE_CACHE,
        // The spatial scaler consumes only a color image and writes the output.
        image_used: SuperResolutionImageUseFlags::SOURCE
            | SuperResolutionImageUseFlags::DESTINATION,
        supported_quality_focuses: SuperResolutionQualityFocusFlags::BALANCED,
        supported_queue_family_indexes: [vk::QUEUE_FAMILY_IGNORED;
            MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT],
        supported_scaling_factor_count,
        supported_scaling_factors,
        max_destination_region_size: vk::Extent2D { width: 16384, height: 16384 },
        max_supported_concurrent_session_dispatches: 1,
    }
}

/// MetalFX scalers allocate and manage their own working memory internally, so
/// a session exposes no Vulkan-visible memory bind points to satisfy.
pub(crate) fn session_memory_requirements() -> Vec<SuperResolutionSessionMemoryRequirements> {
    Vec::new()
}

/// MetalFX manages its own descriptors, so a session requires no
/// application-provided descriptor heap ranges. All sizes are zero.
pub(crate) fn session_descriptor_heap_ranges() -> SuperResolutionDescriptorHeapRanges {
    SuperResolutionDescriptorHeapRanges {
        resource_heap_size: 0,
        resource_heap_alignment: 0,
        sampler_heap_size: 0,
        sampler_heap_alignment: 0,
    }
}

/// A live MetalFX scaler owned by a [`crate::SuperResolutionSession`]. The
/// retained handle keeps the underlying Metal object alive for the session's
/// lifetime.
pub(crate) enum Scaler {
    Temporal(Retained<ProtocolObject<dyn MTL4FXTemporalScaler>>),
    Denoised(Retained<ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>>),
    Spatial(Retained<ProtocolObject<dyn MTL4FXSpatialScaler>>),
}

/// Vulkan-to-Metal pixel format mapping for the formats MetalFX accepts.
const VK_TO_MTL_FORMAT: &[(vk::Format, MTLPixelFormat)] = &[
    (vk::Format::R16G16B16A16_SFLOAT, MTLPixelFormat::RGBA16Float),
    (vk::Format::R8G8B8A8_UNORM, MTLPixelFormat::RGBA8Unorm),
    (vk::Format::B8G8R8A8_UNORM, MTLPixelFormat::BGRA8Unorm),
    (vk::Format::A2B10G10R10_UNORM_PACK32, MTLPixelFormat::RGB10A2Unorm),
    (vk::Format::R32_SFLOAT, MTLPixelFormat::R32Float),
    (vk::Format::D32_SFLOAT, MTLPixelFormat::Depth32Float),
    (vk::Format::R16G16_SFLOAT, MTLPixelFormat::RG16Float),
    (vk::Format::R8_UNORM, MTLPixelFormat::R8Unorm),
    (vk::Format::R16_SFLOAT, MTLPixelFormat::R16Float),
];

/// Maps a Vulkan format to the Metal pixel format MetalFX expects, returning
/// [`MTLPixelFormat::Invalid`] for [`vk::Format::UNDEFINED`] (an absent image)
/// or any unmapped format.
fn mtl_pixel_format(format: vk::Format) -> MTLPixelFormat {
    VK_TO_MTL_FORMAT
        .iter()
        .find(|(vk_format, _)| *vk_format == format)
        .map(|(_, mtl_format)| *mtl_format)
        .unwrap_or(MTLPixelFormat::Invalid)
}

/// Implements [`crate::SuperResolutionSession::new`] for the MetalFX backend.
pub(crate) fn create_session(
    pipeline_cache: &PipelineCache,
    create_info: &SuperResolutionSessionCreateInfo,
) -> VkResult<crate::SuperResolutionSession> {
    let scaler = if create_info.engine == TEMPORAL_SCALER {
        create_temporal_scaler(create_info)?
    } else if create_info.engine == SPATIAL_SCALER {
        create_spatial_scaler(create_info)?
    } else if create_info.engine == TEMPORAL_DENOISED_SCALER {
        create_temporal_denoised_scaler(create_info)?
    } else {
        panic!("unknown super resolution engine for the MetalFX backend")
    };

    Ok(crate::SuperResolutionSession {
        device: pipeline_cache.device().clone(),
        scaler,
    })
}

fn create_temporal_scaler(
    create_info: &SuperResolutionSessionCreateInfo,
) -> VkResult<Scaler> {
    let device = system_default_device();

    // The Metal 4 scaler path requires an `MTL4Compiler`. There is no way to
    // bridge the Vulkan pipeline cache to one, so we create a compiler from the
    // device directly.
    let compiler = device
        .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

    // SAFETY: a freshly created descriptor is fully configured with valid
    // formats and sizes before a scaler is created from it.
    let scaler = unsafe {
        let descriptor = MTLFXTemporalScalerDescriptor::new();
        descriptor.setColorTextureFormat(mtl_pixel_format(create_info.source_format));
        descriptor.setOutputTextureFormat(mtl_pixel_format(create_info.destination_format));
        descriptor.setDepthTextureFormat(mtl_pixel_format(create_info.source_depth_format));
        descriptor.setMotionTextureFormat(mtl_pixel_format(create_info.motion_vector_format));
        descriptor.setInputWidth(create_info.max_source_region_size.width as usize);
        descriptor.setInputHeight(create_info.max_source_region_size.height as usize);
        descriptor.setOutputWidth(create_info.destination_region_size.width as usize);
        descriptor.setOutputHeight(create_info.destination_region_size.height as usize);

        if create_info.reactive_mask_format != vk::Format::UNDEFINED {
            descriptor.setReactiveMaskTextureEnabled(true);
            descriptor
                .setReactiveMaskTextureFormat(mtl_pixel_format(create_info.reactive_mask_format));
        }
        if create_info
            .flags
            .contains(SuperResolutionSessionCreateFlags::USE_AUTO_EXPOSURE)
        {
            descriptor.setAutoExposureEnabled(true);
        }
        if create_info
            .flags
            .contains(SuperResolutionSessionCreateFlags::DYNAMIC_SOURCE_SIZE)
        {
            descriptor.setInputContentPropertiesEnabled(true);
        }
        // `setJitteredMotionVectorsEnabled:` / `setOutputResolutionMotionVectorsEnabled:`
        // are macOS 26/27 additions absent from objc2-metal-fx 0.3.2, so they are
        // sent directly. Both take a `BOOL` and return `void`.
        if create_info
            .flags
            .contains(SuperResolutionSessionCreateFlags::MOTION_VECTORS_USE_JITTER)
        {
            let _: () = msg_send![&*descriptor, setJitteredMotionVectorsEnabled: true];
        }
        if create_info
            .flags
            .contains(SuperResolutionSessionCreateFlags::MOTION_VECTORS_USE_DESTINATION_DIMENSIONS)
        {
            let _: () = msg_send![&*descriptor, setOutputResolutionMotionVectorsEnabled: true];
        }

        descriptor.newTemporalScalerWithDevice_compiler(&device, &compiler)
    };

    // Motion-vector scale is fixed for the session's lifetime.
    if let Some(scaler) = &scaler {
        unsafe {
            scaler.setMotionVectorScaleX(create_info.motion_vector_scale_x);
            scaler.setMotionVectorScaleY(create_info.motion_vector_scale_y);
        }
    }

    scaler
        .map(Scaler::Temporal)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

fn create_temporal_denoised_scaler(
    create_info: &SuperResolutionSessionCreateInfo,
) -> VkResult<Scaler> {
    let device = system_default_device();
    let compiler = device
        .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

    // SAFETY: a freshly created descriptor is fully configured with valid
    // formats and sizes before a scaler is created from it.
    let scaler = unsafe {
        let descriptor = MTLFXTemporalDenoisedScalerDescriptor::new();
        descriptor.setColorTextureFormat(mtl_pixel_format(create_info.source_format));
        descriptor.setOutputTextureFormat(mtl_pixel_format(create_info.destination_format));
        descriptor.setDepthTextureFormat(mtl_pixel_format(create_info.source_depth_format));
        descriptor.setMotionTextureFormat(mtl_pixel_format(create_info.motion_vector_format));
        // The four G-buffer inputs are mandatory for the denoiser.
        descriptor
            .setDiffuseAlbedoTextureFormat(mtl_pixel_format(create_info.diffuse_albedo_format));
        descriptor
            .setSpecularAlbedoTextureFormat(mtl_pixel_format(create_info.specular_albedo_format));
        descriptor.setNormalTextureFormat(mtl_pixel_format(create_info.normal_format));
        descriptor.setRoughnessTextureFormat(mtl_pixel_format(create_info.roughness_format));
        descriptor.setInputWidth(create_info.max_source_region_size.width as usize);
        descriptor.setInputHeight(create_info.max_source_region_size.height as usize);
        descriptor.setOutputWidth(create_info.destination_region_size.width as usize);
        descriptor.setOutputHeight(create_info.destination_region_size.height as usize);

        if create_info.reactive_mask_format != vk::Format::UNDEFINED {
            descriptor.setReactiveMaskTextureEnabled(true);
            descriptor
                .setReactiveMaskTextureFormat(mtl_pixel_format(create_info.reactive_mask_format));
        }
        if create_info.specular_hit_distance_format != vk::Format::UNDEFINED {
            descriptor.setSpecularHitDistanceTextureEnabled(true);
            descriptor.setSpecularHitDistanceTextureFormat(mtl_pixel_format(
                create_info.specular_hit_distance_format,
            ));
        }
        if create_info.denoise_strength_mask_format != vk::Format::UNDEFINED {
            descriptor.setDenoiseStrengthMaskTextureEnabled(true);
            descriptor.setDenoiseStrengthMaskTextureFormat(mtl_pixel_format(
                create_info.denoise_strength_mask_format,
            ));
        }
        if create_info.transparency_overlay_format != vk::Format::UNDEFINED {
            descriptor.setTransparencyOverlayTextureEnabled(true);
            descriptor.setTransparencyOverlayTextureFormat(mtl_pixel_format(
                create_info.transparency_overlay_format,
            ));
        }
        if create_info
            .flags
            .contains(SuperResolutionSessionCreateFlags::USE_AUTO_EXPOSURE)
        {
            descriptor.setAutoExposureEnabled(true);
        }

        descriptor.newTemporalDenoisedScalerWithDevice_compiler(&device, &compiler)
    };

    // Motion-vector scale is fixed for the session's lifetime.
    if let Some(scaler) = &scaler {
        unsafe {
            scaler.setMotionVectorScaleX(create_info.motion_vector_scale_x);
            scaler.setMotionVectorScaleY(create_info.motion_vector_scale_y);
        }
    }

    scaler
        .map(Scaler::Denoised)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

/// Whether a color format carries values beyond the `[0, 1]` range (HDR).
fn is_hdr_format(format: vk::Format) -> bool {
    format == vk::Format::R16G16B16A16_SFLOAT || format == vk::Format::R32G32B32A32_SFLOAT
}

fn create_spatial_scaler(create_info: &SuperResolutionSessionCreateInfo) -> VkResult<Scaler> {
    let device = system_default_device();
    let compiler = device
        .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

    // MetalFX has no dedicated color-space field in our create-info, so it is
    // inferred: an explicit LDR request, or an HDR (float) source format, else
    // the perceptual default.
    let color_processing_mode = if !create_info
        .flags
        .contains(SuperResolutionSessionCreateFlags::FORCE_LDR_COLORS)
        && is_hdr_format(create_info.source_format)
    {
        MTLFXSpatialScalerColorProcessingMode::HDR
    } else {
        MTLFXSpatialScalerColorProcessingMode::Perceptual
    };

    // SAFETY: a freshly created descriptor is fully configured with valid formats
    // and sizes before a scaler is created from it.
    let scaler = unsafe {
        let descriptor = MTLFXSpatialScalerDescriptor::new();
        descriptor.setColorTextureFormat(mtl_pixel_format(create_info.source_format));
        descriptor.setOutputTextureFormat(mtl_pixel_format(create_info.destination_format));
        descriptor.setInputWidth(create_info.max_source_region_size.width as usize);
        descriptor.setInputHeight(create_info.max_source_region_size.height as usize);
        descriptor.setOutputWidth(create_info.destination_region_size.width as usize);
        descriptor.setOutputHeight(create_info.destination_region_size.height as usize);
        descriptor.setColorProcessingMode(color_processing_mode);

        descriptor.newSpatialScalerWithDevice_compiler(&device, &compiler)
    };

    scaler
        .map(Scaler::Spatial)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}

/// Exports the Metal texture backing the image referenced by `image_info`.
///
/// # Safety
///
/// The returned reference borrows a texture owned by the Vulkan implementation;
/// it is valid while the underlying image is alive.
unsafe fn export_texture<'a>(
    device: &'a pumicite::Device,
    image_info: &SuperResolutionImageInfo,
) -> &'a ProtocolObject<dyn MTLTexture> {
    let mut texture_info = vk::ExportMetalTextureInfoEXT::default()
        .image(image_info.view.image)
        .plane(vk::ImageAspectFlags::PLANE_0);
    let mut info = vk::ExportMetalObjectsInfoEXT::default();
    info.p_next = (&mut texture_info as *mut vk::ExportMetalTextureInfoEXT).cast();
    unsafe {
        device
            .extension::<ash::ext::metal_objects::Meta>()
            .export_metal_objects(&mut info);
        &*texture_info
            .mtl_texture
            .cast::<ProtocolObject<dyn MTLTexture>>()
    }
}

/// Clamps a Vulkan offset into the non-negative `NSUInteger` pair MetalFX's
/// content-offset properties expect.
fn offset_xy(offset: vk::Offset2D) -> (usize, usize) {
    (offset.x.max(0) as usize, offset.y.max(0) as usize)
}

/// The `index`-th value of the Halton low-discrepancy sequence for `base`.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Implements [`crate::SuperResolutionSession::recommended_jitter_pattern`].
///
/// MetalFX exposes no jitter API, so this follows Apple's guidance: a
/// Halton(2, 3) sequence whose length scales with the upscale ratio. Each offset
/// is centered into the `[-0.5, 0.5)` texel range that
/// [`MTLFXTemporalScalerBase::jitterOffsetX`] expects.
pub(crate) fn recommended_jitter_pattern(
    session: &crate::SuperResolutionSession,
    destination_region_size: vk::Extent2D,
    source_region_size: vk::Extent2D,
) -> VkResult<Vec<(f32, f32)>> {
    match &session.scaler {
        // Both temporal engines drive jitter the same way.
        Scaler::Temporal(_) | Scaler::Denoised(_) => {
            let scale = destination_region_size.width as f32 / source_region_size.width as f32;
            let phase_count = (8.0 * scale * scale) as u32;
            Ok((1..=phase_count)
                .map(|i| (halton(i, 2) - 0.5, halton(i, 3) - 0.5))
                .collect())
        }
        // The spatial scaler is single-frame and uses no jitter.
        Scaler::Spatial(_) => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
    }
}

/// Implements [`crate::SuperResolutionCommandEncoder::initialize_super_resolution_session`].
///
/// MetalFX scalers are ready to use immediately after creation and have no
/// separate command-buffer initialization step, so this is a no-op.
pub(crate) fn initialize_session(
    _encoder: &mut CommandEncoder,
    _session: &crate::SuperResolutionSession,
) {
}

/// Implements [`crate::SuperResolutionCommandEncoder::dispatch_super_resolution`].
pub(crate) fn dispatch(
    encoder: &mut CommandEncoder,
    session: &crate::SuperResolutionSession,
    dispatch_info: &SuperResolutionDispatchInfo,
) {
    match &session.scaler {
        Scaler::Temporal(scaler) => dispatch_temporal(scaler, encoder, dispatch_info),
        Scaler::Denoised(scaler) => dispatch_denoised(scaler, encoder, dispatch_info),
        Scaler::Spatial(scaler) => dispatch_spatial(scaler, encoder, dispatch_info),
    }
}

fn dispatch_spatial(
    scaler: &ProtocolObject<dyn MTL4FXSpatialScaler>,
    encoder: &CommandEncoder,
    dispatch_info: &SuperResolutionDispatchInfo,
) {
    let device = encoder.device();

    // The spatial scaler exposes no content-offset properties in any macOS
    // version, so image `view_offset`s cannot be applied here.

    // SAFETY: the color and output textures are valid for the duration of this
    // call; the spatial scaler consumes no other inputs.
    unsafe {
        scaler.setColorTexture(Some(export_texture(device, dispatch_info.source_image_info)));
        scaler.setOutputTexture(Some(export_texture(
            device,
            dispatch_info.destination_image_info,
        )));
        scaler.setInputContentWidth(dispatch_info.source_size.width as usize);
        scaler.setInputContentHeight(dispatch_info.source_size.height as usize);

        scaler.encodeToCommandBuffer(encoder.buffer().mtl_command_buffer());
    }
}

fn dispatch_temporal(
    scaler: &ProtocolObject<dyn MTL4FXTemporalScaler>,
    encoder: &CommandEncoder,
    dispatch_info: &SuperResolutionDispatchInfo,
) {
    let device = encoder.device();

    // The `*ContentOffset` / `outputOffset` properties are macOS 26+ additions
    // absent from objc2-metal-fx 0.3.2, so the per-texture region offsets carried
    // by each image info are applied with `msg_send!` (each takes an `NSUInteger`).

    // SAFETY: every Metal object is valid for the duration of this call, and the
    // scaler is fully configured before its work is encoded.
    unsafe {
        scaler.setColorTexture(Some(export_texture(device, dispatch_info.source_image_info)));
        let (x, y) = offset_xy(dispatch_info.source_image_info.view_offset);
        let _: () = msg_send![scaler, setColorContentOffsetX: x];
        let _: () = msg_send![scaler, setColorContentOffsetY: y];

        scaler.setOutputTexture(Some(export_texture(
            device,
            dispatch_info.destination_image_info,
        )));
        let (x, y) = offset_xy(dispatch_info.destination_image_info.view_offset);
        let _: () = msg_send![scaler, setOutputOffsetX: x];
        let _: () = msg_send![scaler, setOutputOffsetY: y];

        if let Some(depth) = dispatch_info.source_depth_image_info {
            scaler.setDepthTexture(Some(export_texture(device, depth)));
            let (x, y) = offset_xy(depth.view_offset);
            let _: () = msg_send![scaler, setDepthContentOffsetX: x];
            let _: () = msg_send![scaler, setDepthContentOffsetY: y];
        }

        if let Some(motion) = dispatch_info.motion_info {
            if let Some(motion_vectors) = motion.motion_vectors_image_info {
                scaler.setMotionTexture(Some(export_texture(device, motion_vectors)));
                let (x, y) = offset_xy(motion_vectors.view_offset);
                let _: () = msg_send![scaler, setMotionContentOffsetX: x];
                let _: () = msg_send![scaler, setMotionContentOffsetY: y];
            }
            if let Some(reactive) = motion.reactive_mask_image_info {
                scaler.setReactiveMaskTexture(Some(export_texture(device, reactive)));
                let (x, y) = offset_xy(reactive.view_offset);
                let _: () = msg_send![scaler, setReactiveMaskContentOffsetX: x];
                let _: () = msg_send![scaler, setReactiveMaskContentOffsetY: y];
            }
            scaler.setJitterOffsetX(motion.texel_jitter_x);
            scaler.setJitterOffsetY(motion.texel_jitter_y);
        }

        if let Some(exposure) = dispatch_info.exposure_info {
            scaler.setPreExposure(exposure.pre_exposure);
            if let Some(exposure_image) = exposure.exposure_scale_image_info {
                scaler.setExposureTexture(Some(export_texture(device, exposure_image)));
            }
        }

        scaler.setInputContentWidth(dispatch_info.source_size.width as usize);
        scaler.setInputContentHeight(dispatch_info.source_size.height as usize);
        scaler.setReset(
            dispatch_info
                .flags
                .contains(SuperResolutionDispatchFlags::RESET_HISTORY),
        );
        scaler.setDepthReversed(
            dispatch_info
                .flags
                .contains(SuperResolutionDispatchFlags::INVERTED_DEPTH_RANGE),
        );

        scaler.encodeToCommandBuffer(encoder.buffer().mtl_command_buffer());
    }
}

fn dispatch_denoised(
    scaler: &ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>,
    encoder: &CommandEncoder,
    dispatch_info: &SuperResolutionDispatchInfo,
) {
    let device = encoder.device();

    // The denoised scaler exposes neither `inputContentWidth/Height` (no dynamic
    // resolution) nor per-texture content-offset properties in any macOS version,
    // so image `view_offset`s cannot be applied here.

    // SAFETY: every Metal object is valid for the duration of this call, and the
    // scaler is fully configured before its work is encoded.
    unsafe {
        scaler.setColorTexture(Some(export_texture(device, dispatch_info.source_image_info)));
        scaler.setOutputTexture(Some(export_texture(
            device,
            dispatch_info.destination_image_info,
        )));

        if let Some(depth) = dispatch_info.source_depth_image_info {
            scaler.setDepthTexture(Some(export_texture(device, depth)));
        }

        if let Some(motion) = dispatch_info.motion_info {
            if let Some(motion_vectors) = motion.motion_vectors_image_info {
                scaler.setMotionTexture(Some(export_texture(device, motion_vectors)));
            }
            if let Some(reactive) = motion.reactive_mask_image_info {
                scaler.setReactiveMaskTexture(Some(export_texture(device, reactive)));
            }
            scaler.setJitterOffsetX(motion.texel_jitter_x);
            scaler.setJitterOffsetY(motion.texel_jitter_y);
        }

        if let Some(exposure) = dispatch_info.exposure_info {
            scaler.setPreExposure(exposure.pre_exposure);
            if let Some(exposure_image) = exposure.exposure_scale_image_info {
                scaler.setExposureTexture(Some(export_texture(device, exposure_image)));
            }
        }

        if let Some(denoise) = dispatch_info.denoise_info {
            scaler.setDiffuseAlbedoTexture(Some(export_texture(
                device,
                denoise.diffuse_albedo_image_info,
            )));
            scaler.setSpecularAlbedoTexture(Some(export_texture(
                device,
                denoise.specular_albedo_image_info,
            )));
            scaler.setNormalTexture(Some(export_texture(device, denoise.normal_image_info)));
            scaler.setRoughnessTexture(Some(export_texture(device, denoise.roughness_image_info)));

            if let Some(specular_hit) = denoise.specular_hit_distance_image_info {
                scaler.setSpecularHitDistanceTexture(Some(export_texture(device, specular_hit)));
            }
            if let Some(strength) = denoise.denoise_strength_mask_image_info {
                scaler.setDenoiseStrengthMaskTexture(Some(export_texture(device, strength)));
            }
            if let Some(overlay) = denoise.transparency_overlay_image_info {
                scaler.setTransparencyOverlayTexture(Some(export_texture(device, overlay)));
            }

            set_denoise_matrices(
                scaler,
                denoise.world_to_view_matrix,
                denoise.view_to_clip_matrix,
            );
        }

        scaler.setShouldResetHistory(
            dispatch_info
                .flags
                .contains(SuperResolutionDispatchFlags::RESET_HISTORY),
        );
        scaler.setDepthReversed(
            dispatch_info
                .flags
                .contains(SuperResolutionDispatchFlags::INVERTED_DEPTH_RANGE),
        );

        scaler.encodeToCommandBuffer(encoder.buffer().mtl_command_buffer());
    }
}

/// Sets the denoiser's world-to-view and view-to-clip matrices.
///
/// `setWorldToViewMatrix:` / `setViewToClipMatrix:` take a `simd_float4x4`, which
/// objc2-metal-fx 0.3.2 does not bind (objc2 has no SIMD support), so they are
/// sent directly. A `simd_float4x4` is a homogeneous aggregate of four
/// `simd_float4` columns, passed in SIMD registers — modeled here with
/// `float32x4_t` so the ABI matches. Implemented for aarch64 (the Apple-silicon
/// target for Metal 4 FX); a no-op elsewhere.
#[cfg(target_arch = "aarch64")]
unsafe fn set_denoise_matrices(
    scaler: &ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>,
    world_to_view: [[f32; 4]; 4],
    view_to_clip: [[f32; 4]; 4],
) {
    use core::arch::aarch64::float32x4_t;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SimdFloat4x4 {
        columns: [float32x4_t; 4],
    }
    // The encoding is metadata only; the call uses the Rust type's ABI.
    unsafe impl objc2::encode::Encode for SimdFloat4x4 {
        const ENCODING: objc2::encode::Encoding = objc2::encode::Encoding::Array(
            4,
            &objc2::encode::Encoding::Array(4, &objc2::encode::Encoding::Float),
        );
    }

    fn to_simd(matrix: [[f32; 4]; 4]) -> SimdFloat4x4 {
        // Each column is four contiguous floats — layout-compatible with float32x4_t.
        SimdFloat4x4 {
            columns: matrix.map(|column| unsafe { core::mem::transmute::<[f32; 4], float32x4_t>(column) }),
        }
    }

    unsafe {
        let _: () = msg_send![scaler, setWorldToViewMatrix: to_simd(world_to_view)];
        let _: () = msg_send![scaler, setViewToClipMatrix: to_simd(view_to_clip)];
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn set_denoise_matrices(
    _scaler: &ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>,
    _world_to_view: [[f32; 4]; 4],
    _view_to_clip: [[f32; 4]; 4],
) {
    // simd_float4x4 ABI handling is implemented for aarch64 only.
}

/// Dispatches [`crate::SuperResolutionPhysicalDevice::get_super_resolution_engine_supported_image_properties`]
/// to the per-engine MetalFX query.
pub(crate) fn engine_supported_image_properties(
    engine: SuperResolutionEngine,
    image_use: SuperResolutionImageUseFlags,
) -> Vec<SuperResolutionImageProperties> {
    if engine == TEMPORAL_SCALER {
        temporal_supported_image_properties(image_use)
    } else if engine == SPATIAL_SCALER {
        spatial_supported_image_properties(image_use)
    } else if engine == TEMPORAL_DENOISED_SCALER {
        denoised_supported_image_properties(image_use)
    } else {
        panic!("unknown super resolution engine for the MetalFX backend")
    }
}

fn denoised_supported_image_properties(
    image_use: SuperResolutionImageUseFlags,
) -> Vec<SuperResolutionImageProperties> {
    use SuperResolutionImageUseFlags as Use;
    use vk::Format;

    const NORMAL_FORMATS: &[Format] =
        &[Format::R16G16B16A16_SFLOAT, Format::A2B10G10R10_UNORM_PACK32];
    const HIT_DISTANCE_FORMATS: &[Format] = &[Format::R16_SFLOAT, Format::R32_SFLOAT];
    let sampled = vk::ImageUsageFlags::SAMPLED;
    let storage = vk::ImageUsageFlags::STORAGE;

    let entries: [(Use, &[Format], vk::ImageUsageFlags); 13] = [
        (Use::SOURCE, TEMPORAL_COLOR_FORMATS, sampled),
        (Use::DESTINATION, TEMPORAL_COLOR_FORMATS, storage),
        (Use::DEPTH, &[Format::R32_SFLOAT, Format::D32_SFLOAT], sampled),
        (Use::MOTION_VECTORS, &[Format::R16G16_SFLOAT], sampled),
        (Use::REACTIVE_MASK, &[Format::R8_UNORM], sampled),
        (Use::EXPOSURE_SCALE, &[Format::R16_SFLOAT], sampled),
        (Use::DIFFUSE_ALBEDO, TEMPORAL_COLOR_FORMATS, sampled),
        (Use::SPECULAR_ALBEDO, TEMPORAL_COLOR_FORMATS, sampled),
        (Use::NORMAL, NORMAL_FORMATS, sampled),
        (Use::ROUGHNESS, &[Format::R8_UNORM, Format::R16_SFLOAT], sampled),
        (Use::SPECULAR_HIT_DISTANCE, HIT_DISTANCE_FORMATS, sampled),
        (Use::DENOISE_STRENGTH_MASK, &[Format::R8_UNORM], sampled),
        (Use::TRANSPARENCY_OVERLAY, &[Format::R16G16B16A16_SFLOAT], sampled),
    ];

    let mut properties = Vec::new();
    for (flag, formats, image_usage_flags) in entries {
        if image_use.contains(flag) {
            properties.extend(formats.iter().map(|&format| SuperResolutionImageProperties {
                format,
                image_usage_flags,
            }));
        }
    }
    properties
}

fn spatial_supported_image_properties(
    image_use: SuperResolutionImageUseFlags,
) -> Vec<SuperResolutionImageProperties> {
    let mut properties = Vec::new();
    if image_use.contains(SuperResolutionImageUseFlags::SOURCE) {
        properties.extend(TEMPORAL_COLOR_FORMATS.iter().map(|&format| {
            SuperResolutionImageProperties {
                format,
                image_usage_flags: vk::ImageUsageFlags::SAMPLED,
            }
        }));
    }
    if image_use.contains(SuperResolutionImageUseFlags::DESTINATION) {
        properties.extend(TEMPORAL_COLOR_FORMATS.iter().map(|&format| {
            SuperResolutionImageProperties {
                format,
                image_usage_flags: vk::ImageUsageFlags::STORAGE,
            }
        }));
    }
    properties
}

fn temporal_supported_image_properties(
    image_use: SuperResolutionImageUseFlags,
) -> Vec<SuperResolutionImageProperties> {
    let mut properties = Vec::new();
    for (flag, role) in [
        (SuperResolutionImageUseFlags::SOURCE, TemporalTextureRole::Color),
        (SuperResolutionImageUseFlags::DESTINATION, TemporalTextureRole::Output),
        (SuperResolutionImageUseFlags::DEPTH, TemporalTextureRole::Depth),
        (SuperResolutionImageUseFlags::MOTION_VECTORS, TemporalTextureRole::Motion),
        (SuperResolutionImageUseFlags::REACTIVE_MASK, TemporalTextureRole::ReactiveMask),
    ] {
        if image_use.contains(flag) {
            let (formats, image_usage_flags) = role.supported();
            properties.extend(formats.iter().map(|&format| {
                SuperResolutionImageProperties { format, image_usage_flags }
            }));
        }
    }

    // The exposure scale is a fixed 1x1 R16Float texture MetalFX samples; it has
    // no descriptor field or usage getter, so it is reported directly.
    if image_use.contains(SuperResolutionImageUseFlags::EXPOSURE_SCALE) {
        properties.push(SuperResolutionImageProperties {
            format: vk::Format::R16_SFLOAT,
            image_usage_flags: vk::ImageUsageFlags::SAMPLED,
        });
    }

    properties
}
