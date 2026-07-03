//! Plain-Vulkan blit implementation of the super-resolution interface.
//!
//! [`SuperResolutionEngine::VULKAN_BLIT`] upscales with a single
//! `vkCmdBlitImage` (linear filter). It consumes only a source color image and
//! writes the destination — no temporal accumulation, no denoising, no jitter.
//! It exists as a universally available fallback for platforms/drivers where
//! no vendor engine works, and as a reference backend for bring-up debugging.

use core::ffi::c_char;

use ash::{VkResult, vk};
use pumicite::HasDevice;
use pumicite::command::CommandEncoder;
use pumicite::physical_device::PhysicalDevice;
use pumicite::utils::AsVkHandle;

use crate::{
    MAX_SUPER_RESOLUTION_NAME_SIZE, MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT,
    MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT, ScalingFactor, SuperResolutionDispatchInfo,
    SuperResolutionEngineProperties, SuperResolutionEnginePropertyFlags,
    SuperResolutionImageProperties, SuperResolutionImageUseFlags, SuperResolutionQualityFocusFlags,
    SuperResolutionSessionCreateInfo,
};

/// Color formats accepted for both the source and the destination. All of
/// these support `BLIT_SRC`/`BLIT_DST` with linear filtering on every Vulkan
/// implementation dust targets.
const COLOR_FORMATS: &[vk::Format] = &[
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::B8G8R8A8_UNORM,
    vk::Format::A2B10G10R10_UNORM_PACK32,
];

/// Representative discrete scaling factors; blit scaling is fully continuous.
const SCALING_FACTORS: &[ScalingFactor] = &[
    ScalingFactor {
        numerator: 1,
        denominator: 1,
    }, // 1.00x (copy)
    ScalingFactor {
        numerator: 3,
        denominator: 2,
    }, // 1.50x
    ScalingFactor {
        numerator: 5,
        denominator: 3,
    }, // 1.67x
    ScalingFactor {
        numerator: 7,
        denominator: 4,
    }, // 1.75x
    ScalingFactor {
        numerator: 2,
        denominator: 1,
    }, // 2.00x
    ScalingFactor {
        numerator: 5,
        denominator: 2,
    }, // 2.50x
    ScalingFactor {
        numerator: 3,
        denominator: 1,
    }, // 3.00x
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

/// Implements [`crate::SuperResolutionPhysicalDevice::get_super_resolution_engine_properties`]
/// for [`crate::SuperResolutionEngine::VULKAN_BLIT`].
pub(crate) fn engine_properties(
    physical_device: &PhysicalDevice,
) -> SuperResolutionEngineProperties {
    let mut supported_scaling_factors = [ScalingFactor {
        numerator: 0,
        denominator: 1,
    }; MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT];
    supported_scaling_factors[..SCALING_FACTORS.len()].copy_from_slice(SCALING_FACTORS);

    let max_dimension = physical_device.properties().limits.max_image_dimension2_d;

    SuperResolutionEngineProperties {
        vendor_id: physical_device.properties().vendor_id,
        engine_version: 0,
        engine_uuid: [0u8; vk::UUID_SIZE],
        engine_name: engine_name(b"Vulkan Blit"),
        // Single-frame: not temporal. Any source/destination size combination
        // works, and the per-dispatch `source_size` makes the source dynamic.
        flags: SuperResolutionEnginePropertyFlags::SUPPORTS_CONTINUOUS_SCALING
            | SuperResolutionEnginePropertyFlags::SUPPORTS_DYNAMIC_SOURCE_SIZE,
        image_used: SuperResolutionImageUseFlags::SOURCE
            | SuperResolutionImageUseFlags::DESTINATION,
        // A blit has no quality/performance trade-off to make; accept any focus.
        supported_quality_focuses: SuperResolutionQualityFocusFlags::BALANCED
            | SuperResolutionQualityFocusFlags::QUALITY
            | SuperResolutionQualityFocusFlags::PERFORMANCE
            | SuperResolutionQualityFocusFlags::POWER,
        supported_queue_family_indexes: [vk::QUEUE_FAMILY_IGNORED;
            MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT],
        supported_scaling_factor_count: SCALING_FACTORS.len() as u32,
        supported_scaling_factors,
        max_destination_region_size: vk::Extent2D {
            width: max_dimension,
            height: max_dimension,
        },
        max_supported_concurrent_session_dispatches: u32::MAX,
    }
}

/// Implements [`crate::SuperResolutionPhysicalDevice::get_super_resolution_engine_supported_image_properties`]
/// for [`crate::SuperResolutionEngine::VULKAN_BLIT`].
pub(crate) fn engine_supported_image_properties(
    image_use: SuperResolutionImageUseFlags,
) -> Vec<SuperResolutionImageProperties> {
    let mut properties = Vec::new();
    if image_use.contains(SuperResolutionImageUseFlags::SOURCE) {
        properties.extend(
            COLOR_FORMATS
                .iter()
                .map(|&format| SuperResolutionImageProperties {
                    format,
                    image_usage_flags: vk::ImageUsageFlags::TRANSFER_SRC,
                }),
        );
    }
    if image_use.contains(SuperResolutionImageUseFlags::DESTINATION) {
        properties.extend(
            COLOR_FORMATS
                .iter()
                .map(|&format| SuperResolutionImageProperties {
                    format,
                    image_usage_flags: vk::ImageUsageFlags::TRANSFER_DST,
                }),
        );
    }
    properties
}

/// State backing a blit [`crate::SuperResolutionSession`]. A blit is stateless;
/// only the destination region size (not part of the dispatch info) is kept.
pub(crate) struct BlitSession {
    destination_region_size: vk::Extent2D,
}

/// Implements [`crate::SuperResolutionSession::new`] for the blit backend.
pub(crate) fn create_session(
    create_info: &SuperResolutionSessionCreateInfo,
) -> VkResult<BlitSession> {
    if !COLOR_FORMATS.contains(&create_info.source_format)
        || !COLOR_FORMATS.contains(&create_info.destination_format)
    {
        return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
    }
    Ok(BlitSession {
        destination_region_size: create_info.destination_region_size,
    })
}

/// Maps an image-info subresource range to the single-mip layers a blit region
/// addresses.
fn subresource_layers(range: &vk::ImageSubresourceRange) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: range.aspect_mask,
        mip_level: range.base_mip_level,
        base_array_layer: range.base_array_layer,
        layer_count: if range.layer_count == vk::REMAINING_ARRAY_LAYERS {
            1
        } else {
            range.layer_count
        },
    }
}

/// Implements [`crate::SuperResolutionCommandEncoder::dispatch_super_resolution`]
/// for the blit backend: a single linear-filtered `vkCmdBlitImage` from the
/// source region to the destination region. All other dispatch inputs (depth,
/// motion, exposure, G-buffers) are ignored.
pub(crate) fn dispatch(
    encoder: &mut CommandEncoder,
    session: &BlitSession,
    dispatch_info: &SuperResolutionDispatchInfo,
) {
    let src = dispatch_info.source_image_info;
    let dst = dispatch_info.destination_image_info;
    let src_size = dispatch_info.source_size;
    let dst_size = session.destination_region_size;

    let region = vk::ImageBlit {
        src_subresource: subresource_layers(&src.view.subresource_range),
        src_offsets: [
            vk::Offset3D {
                x: src.view_offset.x,
                y: src.view_offset.y,
                z: 0,
            },
            vk::Offset3D {
                x: src.view_offset.x + src_size.width as i32,
                y: src.view_offset.y + src_size.height as i32,
                z: 1,
            },
        ],
        dst_subresource: subresource_layers(&dst.view.subresource_range),
        dst_offsets: [
            vk::Offset3D {
                x: dst.view_offset.x,
                y: dst.view_offset.y,
                z: 0,
            },
            vk::Offset3D {
                x: dst.view_offset.x + dst_size.width as i32,
                y: dst.view_offset.y + dst_size.height as i32,
                z: 1,
            },
        ],
    };

    // SAFETY: the image handles originate from the caller's live resources,
    // which must outlive this command buffer's GPU execution, and the caller is
    // responsible for having transitioned them to `initial_layout`.
    unsafe {
        encoder.device().cmd_blit_image(
            encoder.buffer().vk_handle(),
            src.view.image,
            src.initial_layout,
            dst.view.image,
            dst.initial_layout,
            &[region],
            vk::Filter::LINEAR,
        );
    }
}
