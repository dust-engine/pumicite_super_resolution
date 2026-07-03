use ash::{VkResult, vk};
use core::ffi::c_char;
use pumicite::Device;
use pumicite::HasDevice;
use pumicite::command::CommandEncoder;
use pumicite::physical_device::PhysicalDevice;
use pumicite::pipeline::PipelineCache;
use pumicite::utils::AsVkHandle;
use std::ffi::{CStr, c_int};
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};

use crate::{
    MAX_SUPER_RESOLUTION_NAME_SIZE, MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT,
    MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT, ScalingFactor, SuperResolutionDescriptorHeapRanges,
    SuperResolutionDispatchFlags, SuperResolutionDispatchInfo, SuperResolutionEngine,
    SuperResolutionEngineProperties, SuperResolutionEnginePropertyFlags, SuperResolutionImageInfo,
    SuperResolutionImageProperties, SuperResolutionImageUseFlags, SuperResolutionQualityFocusFlags,
    SuperResolutionSessionCreateFlags, SuperResolutionSessionCreateInfo,
    SuperResolutionSessionMemoryRequirements,
};

pub mod sys;

// ============================================================================
// Result helpers
// ============================================================================

impl sys::NVSDK_NGX_Result {
    #[inline]
    pub fn result(self) -> DlssResult<()> {
        self.result_with_success(())
    }

    #[inline]
    pub fn result_with_success<T>(self, v: T) -> DlssResult<T> {
        match self {
            Self::Success => Ok(v),
            _ => Err(self),
        }
    }

    /// Symbolic name from `nvsdk_ngx_defs.h`, or `None` for unrecognised codes.
    fn name(self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::Fail => "Fail",
            Self::FAIL_FeatureNotSupported => "FAIL_FeatureNotSupported",
            Self::FAIL_PlatformError => "FAIL_PlatformError",
            Self::FAIL_FeatureAlreadyExists => "FAIL_FeatureAlreadyExists",
            Self::FAIL_FeatureNotFound => "FAIL_FeatureNotFound",
            Self::FAIL_InvalidParameter => "FAIL_InvalidParameter",
            Self::FAIL_ScratchBufferTooSmall => "FAIL_ScratchBufferTooSmall",
            Self::FAIL_NotInitialized => "FAIL_NotInitialized",
            Self::FAIL_UnsupportedInputFormat => "FAIL_UnsupportedInputFormat",
            Self::FAIL_RWFlagMissing => "FAIL_RWFlagMissing",
            Self::FAIL_MissingInput => "FAIL_MissingInput",
            Self::FAIL_UnableToInitializeFeature => "FAIL_UnableToInitializeFeature",
            Self::FAIL_OutOfDate => "FAIL_OutOfDate",
            Self::FAIL_OutOfGPUMemory => "FAIL_OutOfGPUMemory",
            Self::FAIL_UnsupportedFormat => "FAIL_UnsupportedFormat",
            Self::FAIL_UnableToWriteToAppDataPath => "FAIL_UnableToWriteToAppDataPath",
            Self::FAIL_UnsupportedParameter => "FAIL_UnsupportedParameter",
            Self::FAIL_Denied => "FAIL_Denied",
            Self::FAIL_NotImplemented => "FAIL_NotImplemented",
            _ => "FAIL_Unknown",
        }
    }
}

impl core::fmt::Debug for sys::NVSDK_NGX_Result {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NVSDK_NGX_Result::{}(0x{:08x})", self.name(), self.0)
    }
}

impl core::fmt::Display for sys::NVSDK_NGX_Result {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NVSDK_NGX_Result::{}(0x{:08x})", self.name(), self.0)
    }
}

impl core::error::Error for sys::NVSDK_NGX_Result {}

pub type DlssResult<T> = core::result::Result<T, sys::NVSDK_NGX_Result>;

// ============================================================================
// Parameter map
// ============================================================================

pub struct ParameterMap(NonNull<sys::NVSDK_NGX_Parameter>);

impl Drop for ParameterMap {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::NVSDK_NGX_VULKAN_DestroyParameters(self.0.as_ptr()).result() {
                tracing::warn!(target: "ngx", "DestroyParameters failed: {e}");
            }
        }
    }
}

impl sys::NVSDK_NGX_FeatureDiscoveryInfo {
    /// `app_data_path`, `project_id`, and `engine_version` must all remain alive
    /// for as long as the returned struct is used by NGX (it stores pointers into
    /// them). `app_data_path` must be NUL-terminated.
    pub fn new(
        app_data_path: &[sys::wchar_t],
        project_id: &CStr,
        engine_version: &CStr,
    ) -> Self {
        debug_assert!(
            app_data_path.last() == Some(&0),
            "ApplicationDataPath must be NUL-terminated"
        );
        sys::NVSDK_NGX_FeatureDiscoveryInfo {
            SDKVersion: sys::NVSDK_NGX_VERSION_API,
            FeatureID: sys::NVSDK_NGX_Feature::RayReconstruction,
            Identifier: sys::NVSDK_NGX_Application_Identifier {
                IdentifierType: sys::NVSDK_NGX_Application_Identifier_Type::ProjectId,
                v: sys::NVSDK_NGX_Application_Identifier_Value {
                    ProjectDesc: sys::NVSDK_NGX_ProjectIdDescription {
                        ProjectId: project_id.as_ptr(),
                        EngineType: sys::NVSDK_NGX_EngineType::Custom,
                        EngineVersion: engine_version.as_ptr(),
                    },
                },
            },
            ApplicationDataPath: app_data_path.as_ptr(),
            FeatureInfo: std::ptr::null(),
        }
    }
}

// ============================================================================
// Process-global NGX runtime
// ============================================================================

/// The live NGX runtime: owns the `VkDevice` NGX was initialized with and runs
/// `NVSDK_NGX_VULKAN_Shutdown` on drop. Shared via [`Arc`] with every
/// [`NgxFeature`] and live [`DlssSession`] so Shutdown only fires once the last
/// one is released.
struct NgxRuntime {
    device: Device,
}

impl Drop for NgxRuntime {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::NVSDK_NGX_VULKAN_Shutdown(self.device.vk_handle()).result() {
                tracing::warn!(target: "ngx", "Shutdown failed: {e}");
            }
        }
    }
}

/// Process-global NGX runtime. NGX is a non-thread-safe per-process singleton;
/// the lock guards init-once and `Arc` hand-out. `None` until [`ensure_runtime`]
/// initializes it on first use. Serialization of the NGX entry points
/// themselves relies on the renderer driving them from one submission thread.
static NGX: Mutex<Option<Arc<NgxRuntime>>> = Mutex::new(None);

/// Returns the global runtime, or `FAIL_NotInitialized` if [`ensure_runtime`]
/// hasn't initialized it yet.
fn runtime() -> DlssResult<Arc<NgxRuntime>> {
    NGX.lock()
        .unwrap()
        .clone()
        .ok_or(sys::NVSDK_NGX_Result::FAIL_NotInitialized)
}

/// Returns the global NGX runtime, initializing it (`NVSDK_NGX_VULKAN_Init`) on
/// first use with `device`. NGX Init requires a logical `VkDevice`, so this is
/// the earliest point the runtime can exist; feature support was already
/// confirmed at engine enumeration via the pre-init discovery API.
fn ensure_runtime(
    device: &Device,
    application_info: &crate::SuperResolutionApplicationInfo<'_>,
) -> DlssResult<Arc<NgxRuntime>> {
    let mut guard = NGX.lock().unwrap();
    if let Some(runtime) = guard.as_ref() {
        return Ok(runtime.clone());
    }

    let app_info = application_info;
    let app_data = encode_app_data_path(app_info.application_data_path);

    // Tell NGX where to find the DLSS-RR runtime library. The standalone Cargo
    // build (build.rs) bakes `DLSS_RUNTIME_DIR`; a Bazel build bundles it in
    // runfiles. If unset, NGX falls back to searching the application directory.
    let dll_dir = dlss_runtime_search_dir().map(|p| encode_app_data_path(&p));
    let mut dll_dir_ptr = dll_dir.as_ref().map(|x| x.as_ptr()).unwrap_or(ptr::null());

    unsafe {
        sys::NVSDK_NGX_VULKAN_Init_with_ProjectID(
            app_info.project_id.as_ptr(),
            sys::NVSDK_NGX_EngineType::Custom,
            app_info.engine_version.as_ptr(),
            app_data.as_ptr(),
            device.instance().handle(),
            device.physical_device().vk_handle(),
            device.vk_handle(),
            Some(device.instance().entry().static_fn().get_instance_proc_addr),
            Some(device.instance().fp_v1_0().get_device_proc_addr),
            &sys::NVSDK_NGX_FeatureCommonInfo {
                PathListInfo: if dll_dir_ptr.is_null() {
                    sys::NVSDK_NGX_PathListInfo::default()
                } else {
                    sys::NVSDK_NGX_PathListInfo {
                        Path: &mut dll_dir_ptr,
                        Length: 1,
                    }
                },
                InternalData: std::ptr::null_mut(),
                LoggingInfo: sys::NVSDK_NGX_LoggingInfo {
                    LoggingCallback: Some(ngx_log_callback),
                    MinimumLoggingLevel: sys::NVSDK_NGX_Logging_Level::On,
                    DisableOtherLoggingSinks: true,
                },
            },
            sys::NVSDK_NGX_VERSION_API,
        )
        .result()?;
    }

    let runtime = Arc::new(NgxRuntime {
        device: device.clone(),
    });
    *guard = Some(runtime.clone());
    Ok(runtime)
}

/// Bazel runfiles path of the DLSS-RR runtime library. The leading `dlss` is the
/// *apparent* repo name (this crate's `MODULE.bazel` declares `@dlss`), mapped to
/// its canonical repo by [`Runfiles::rlocation_from`] using the calling module's
/// repo mapping.
#[cfg(target_os = "windows")]
const DLSSD_RUNFILES_PATH: &str = "dlss/runtime/nvngx_dlssd.dll";
#[cfg(target_os = "linux")]
const DLSSD_RUNFILES_PATH: &str = "dlss/runtime/libnvidia-ngx-dlssd.so.310.6.0";

/// Directory containing the DLSS-RR runtime library (`nvngx_dlssd.dll` /
/// `libnvidia-ngx-dlssd.so`), used as an NGX search path. If `None`, NGX falls
/// back to searching the application directory.
///
/// Resolution order: `DLSS_RUNTIME_DIR` baked at compile time by the standalone
/// Cargo build (`build.rs`); then `DLSS_RUNTIME_DIR` from the runtime
/// environment; then Bazel runfiles (the DLL is bundled there on the Bazel path).
fn dlss_runtime_search_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = option_env!("DLSS_RUNTIME_DIR") {
        if !dir.is_empty() {
            return Some(std::path::PathBuf::from(dir));
        }
    }
    if let Ok(dir) = std::env::var("DLSS_RUNTIME_DIR") {
        if !dir.is_empty() {
            return Some(std::path::PathBuf::from(dir));
        }
    }
    // `rlocation_from` maps the apparent `dlss` repo via the calling module's
    // repo mapping. The source repo comes from `REPOSITORY_NAME`, which
    // rules_rust sets under Bazel and is absent under standalone Cargo (where
    // the env paths above apply, so this lookup is skipped / returns None).
    if let Ok(runfiles) = runfiles::Runfiles::create() {
        let source_repo = option_env!("REPOSITORY_NAME").unwrap_or("");
        if let Some(path) = runfiles.rlocation_from(DLSSD_RUNFILES_PATH, source_repo) {
            return path.parent().map(|p| p.to_path_buf());
        }
    }
    None
}

// ============================================================================
// NGX operations (free functions over the global runtime)
// ============================================================================

/// Allocates an empty NGX parameter map. The runtime must be initialized.
fn allocate_parameters() -> DlssResult<ParameterMap> {
    let mut parameters: *mut sys::NVSDK_NGX_Parameter = ptr::null_mut();
    unsafe {
        sys::NVSDK_NGX_VULKAN_AllocateParameters(&mut parameters).result()?;
    }
    Ok(ParameterMap(NonNull::new(parameters).unwrap()))
}

/// Reports whether DLSS-RR is supported on `physical_device`, using NGX's
/// pre-init discovery API (`NVSDK_NGX_VULKAN_GetFeatureRequirements`) — no
/// logical `VkDevice` / `NVSDK_NGX_VULKAN_Init` required.
///
/// Called at engine enumeration. Because a [`SuperResolutionEngine`] for DLSS is
/// only ever handed back when this returned `true`, downstream operations
/// (`create_session`, `engine_properties`, `dispatch`) can assume availability
/// without re-checking.
fn check_dlss_rr_available(
    physical_device: &PhysicalDevice,
    app_info: &crate::SuperResolutionApplicationInfo<'_>,
) -> bool {
    let app_data = encode_app_data_path(app_info.application_data_path);

    // Point NGX at the DLSS-RR runtime library so discovery can inspect the
    // snippet's requirements (best-effort; it also checks driver/HW support).
    let dll_dir = dlss_runtime_search_dir().map(|p| encode_app_data_path(&p));
    let mut dll_dir_ptr = dll_dir.as_ref().map(|x| x.as_ptr()).unwrap_or(ptr::null());
    let feature_info = sys::NVSDK_NGX_FeatureCommonInfo {
        PathListInfo: if dll_dir_ptr.is_null() {
            sys::NVSDK_NGX_PathListInfo::default()
        } else {
            sys::NVSDK_NGX_PathListInfo {
                Path: &mut dll_dir_ptr,
                Length: 1,
            }
        },
        InternalData: ptr::null_mut(),
        LoggingInfo: sys::NVSDK_NGX_LoggingInfo {
            LoggingCallback: None,
            MinimumLoggingLevel: sys::NVSDK_NGX_Logging_Level::Off,
            DisableOtherLoggingSinks: true,
        },
    };
    let mut info = sys::NVSDK_NGX_FeatureDiscoveryInfo::new(
        &app_data,
        app_info.project_id,
        app_info.engine_version,
    );
    info.FeatureInfo = &feature_info;

    let mut requirement = MaybeUninit::<sys::NVSDK_NGX_FeatureRequirement>::uninit();
    // SAFETY: instance + physical device are valid; `info` and everything it
    // points at outlive the call; NGX writes `requirement` only on success.
    let result = unsafe {
        sys::NVSDK_NGX_VULKAN_GetFeatureRequirements(
            physical_device.instance().handle(),
            physical_device.vk_handle(),
            &info,
            requirement.as_mut_ptr(),
        )
    };
    if result.result().is_err() {
        return false;
    }
    unsafe { requirement.assume_init() }.FeatureSupported
        == sys::NVSDK_NGX_FeatureSupportResult_Supported
}

pub(crate) fn required_instance_extensions(
    instance_builder: &mut pumicite::instance::InstanceBuilder,
    app_info: &crate::SuperResolutionApplicationInfo<'_>,
) -> VkResult<()> {
    let app_data = encode_app_data_path(app_info.application_data_path);
    let info = sys::NVSDK_NGX_FeatureDiscoveryInfo::new(
        &app_data,
        app_info.project_id,
        app_info.engine_version,
    );
    let mut count: u32 = 0;
    let mut properties: *mut vk::ExtensionProperties = ptr::null_mut();
    // SAFETY: `info` (and everything it points at) outlives the call; NGX writes
    // `count`/`properties` only on success and the returned array is static.
    let result = unsafe {
        sys::NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            &info,
            &mut count,
            &mut properties,
        )
    };
    if result.result().is_err() || properties.is_null() {
        return Ok(());
    }
    let props = unsafe { std::slice::from_raw_parts(properties, count as usize) };
    for extension in props {
        instance_builder.enable_extension_named(extension.extension_name_as_c_str().unwrap())?;
    }
    Ok(())
}

pub(crate) fn required_device_extensions(
    device_builder: &mut pumicite::device::DeviceBuilder,
    app_info: &crate::SuperResolutionApplicationInfo<'_>,
) -> VkResult<()> {
    let driver = device_builder
        .physical_device()
        .properties()
        .get::<vk::PhysicalDeviceDriverProperties>();
    if driver.driver_id != vk::DriverId::NVIDIA_PROPRIETARY {
        return Ok(());
    }
    let app_data = encode_app_data_path(app_info.application_data_path);
    let info = sys::NVSDK_NGX_FeatureDiscoveryInfo::new(
        &app_data,
        app_info.project_id,
        app_info.engine_version,
    );
    let mut count: u32 = 0;
    let mut properties: *mut vk::ExtensionProperties = ptr::null_mut();
    // SAFETY: instance + physical device are valid; `info` outlives the call;
    // NGX writes `count`/`properties` only on success and the array is static.
    let result = unsafe {
        sys::NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            device_builder.physical_device().instance().handle(),
            device_builder.physical_device().vk_handle(),
            &info,
            &mut count,
            &mut properties,
        )
    };
    if result.result().is_err() || properties.is_null() {
        tracing::warn!(target: "ngx", "GetFeatureDeviceExtensionRequirements failed: {result:?}");
        return Ok(());
    }
    let props = unsafe { std::slice::from_raw_parts(properties, count as usize) };
    for extension in props {
        device_builder.enable_extension_named(extension.extension_name_as_c_str().unwrap())?;
    }
    Ok(())
}

/// Creates a DLSS-RR feature on `cmd_buffer` (which must be recording).
fn create_dlssd_feature(
    cmd_buffer: vk::CommandBuffer,
    create_params: &sys::NVSDK_NGX_DLSSD_Create_Params,
) -> DlssResult<NgxFeature> {
    let runtime = runtime()?;
    let params = allocate_parameters()?;
    let mut handle: *mut sys::NVSDK_NGX_Handle = ptr::null_mut();
    unsafe {
        sys::pumicite_sr_ngx_vulkan_create_dlssd_ext1(
            runtime.device.vk_handle(),
            cmd_buffer,
            1, // Multi-GPU creation node mask (default 1)
            1, // Multi-GPU visibility node mask (default 1)
            &mut handle,
            params.0.as_ptr(),
            create_params,
        )
        .result()?;
    }

    Ok(NgxFeature {
        handle: NonNull::new(handle).expect("NGX returned null handle on success"),
        _runtime: runtime,
    })
}

/// Records a DLSS-RR evaluate dispatch into `cmd_buffer`.
fn evaluate_dlssd(
    cmd_buffer: vk::CommandBuffer,
    feature: &NgxFeature,
    eval_params: &mut sys::NVSDK_NGX_VK_DLSSD_Eval_Params,
) -> DlssResult<()> {
    let _runtime = runtime()?;
    let params = allocate_parameters()?;
    unsafe {
        sys::pumicite_sr_ngx_vulkan_evaluate_dlssd_ext(
            cmd_buffer,
            feature.handle.as_ptr(),
            params.0.as_ptr(),
            eval_params,
        )
        .result()
    }
}

/// Owns an NGX feature handle (a DLSS-RR instance). Drop calls
/// `NVSDK_NGX_VULKAN_ReleaseFeature`, then releases its strong reference to the
/// shared runtime.
pub struct NgxFeature {
    handle: NonNull<sys::NVSDK_NGX_Handle>,
    // Dropped after `handle` (ReleaseFeature), so Shutdown can only fire once
    // the last feature/session reference is gone.
    _runtime: Arc<NgxRuntime>,
}

unsafe impl Send for NgxFeature {}
unsafe impl Sync for NgxFeature {}

impl Drop for NgxFeature {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = sys::NVSDK_NGX_VULKAN_ReleaseFeature(self.handle.as_ptr()).result() {
                tracing::warn!(target: "ngx", "ReleaseFeature failed: {e}");
            }
        }
    }
}

// ============================================================================
// NGX resource wrapper
// ============================================================================

/// Wraps an [`sys::NVSDK_NGX_Resource_VK`] describing a Vulkan image view NGX
/// reads from / writes to during evaluation. NGX dereferences the pointer
/// inside `EvaluateFeature`, so the underlying image/view/memory must stay alive
/// until the recorded command buffer completes.
#[repr(transparent)]
pub struct Resource(sys::NVSDK_NGX_Resource_VK);

impl Resource {
    /// Builds a resource backed by a Vulkan image view. `read_write` must be
    /// `true` for output / mutated targets and `false` for read-only inputs.
    pub fn image_view(
        image_view: vk::ImageView,
        image: vk::Image,
        subresource_range: vk::ImageSubresourceRange,
        format: vk::Format,
        width: u32,
        height: u32,
        read_write: bool,
    ) -> Self {
        Self(sys::NVSDK_NGX_Resource_VK {
            Resource: sys::NVSDK_NGX_Resource_VK_Resource {
                ImageViewInfo: sys::NVSDK_NGX_ImageViewInfo_VK {
                    ImageView: image_view,
                    Image: image,
                    SubresourceRange: subresource_range,
                    Format: format,
                    Width: width,
                    Height: height,
                },
            },
            Type: sys::NVSDK_NGX_Resource_VK_Type::ImageView,
            ReadWrite: read_write,
        })
    }

    fn as_mut_ptr(&mut self) -> *mut sys::NVSDK_NGX_Resource_VK {
        &mut self.0
    }
}

unsafe extern "C" fn ngx_log_callback(
    message: *const core::ffi::c_char,
    level: sys::NVSDK_NGX_Logging_Level,
    component: sys::NVSDK_NGX_Feature,
) {
    if message.is_null() {
        return;
    }
    let msg = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let msg = msg.trim_end_matches(['\r', '\n']);
    match level {
        sys::NVSDK_NGX_Logging_Level::Off => {}
        sys::NVSDK_NGX_Logging_Level::On => {
            tracing::info!(target: "ngx", ?component, "{msg}")
        }
        sys::NVSDK_NGX_Logging_Level::Verbose => {
            tracing::debug!(target: "ngx", ?component, "{msg}")
        }
    }
}

/// NUL-terminated `wchar_t` encoding of `path`, suitable for NGX's
/// `ApplicationDataPath` / `PathListInfo`.
pub(crate) fn encode_app_data_path(path: &std::path::Path) -> Vec<sys::wchar_t> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let mut v: Vec<u16> = path.as_os_str().encode_wide().collect();
        v.push(0);
        v
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let mut v: Vec<sys::wchar_t> = path
            .as_os_str()
            .as_bytes()
            .iter()
            .map(|&b| b as sys::wchar_t)
            .collect();
        v.push(0);
        v
    }
}

// ============================================================================
// Generic super-resolution backend
// ============================================================================

/// Engines provided by the DLSS backend, in enumeration order.
pub(crate) const ENGINES: [SuperResolutionEngine; 1] = [SuperResolutionEngine::DLSS_RR];

/// Output-over-input ratios exposed for DLSS-RR, matching NVIDIA's quality
/// modes so a ratio-derived render extent equals NGX's recommended optimal:
/// DLAA (1.0), UltraQuality (~0.77), Quality (0.667), Balanced (~0.58),
/// Performance (0.5), UltraPerformance (~0.33).
const SCALING_FACTORS: [ScalingFactor; 6] = [
    ScalingFactor { numerator: 1, denominator: 1 }, // 1.00x — DLAA
    ScalingFactor { numerator: 13, denominator: 10 }, // 1.30x — UltraQuality
    ScalingFactor { numerator: 3, denominator: 2 }, // 1.50x — Quality
    ScalingFactor { numerator: 12, denominator: 7 }, // 1.71x — Balanced
    ScalingFactor { numerator: 2, denominator: 1 }, // 2.00x — Performance
    ScalingFactor { numerator: 3, denominator: 1 }, // 3.00x — UltraPerformance
];

/// Color formats DLSS-RR accepts for the source color and upscaled output.
const COLOR_FORMATS: &[vk::Format] =
    &[vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8G8B8A8_UNORM];

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

/// Enumerates the DLSS backend's engines for `physical_device`. DLSS-RR is
/// reported only on NVIDIA's proprietary driver *and* when NGX's pre-init
/// discovery API confirms the feature is supported (driver new enough, RR
/// snippet present). This is the single availability gate: any DLSS
/// [`SuperResolutionEngine`] handed out implies it passed here.
pub(crate) fn enumerate_engines(
    physical_device: &PhysicalDevice,
    application_info: &crate::SuperResolutionApplicationInfo<'_>,
) -> Vec<SuperResolutionEngine> {
    let driver = physical_device
        .properties()
        .get::<vk::PhysicalDeviceDriverProperties>();
    if driver.driver_id == vk::DriverId::NVIDIA_PROPRIETARY
        && check_dlss_rr_available(physical_device, application_info)
    {
        ENGINES.to_vec()
    } else {
        Vec::new()
    }
}

pub(crate) fn engine_properties(
    physical_device: &PhysicalDevice,
    engine: SuperResolutionEngine,
) -> SuperResolutionEngineProperties {
    assert!(
        engine == SuperResolutionEngine::DLSS_RR,
        "unknown super resolution engine for the DLSS backend"
    );

    let mut supported_scaling_factors =
        [ScalingFactor { numerator: 0, denominator: 1 }; MAX_SUPER_RESOLUTION_SCALING_FACTOR_COUNT];
    supported_scaling_factors[..SCALING_FACTORS.len()].copy_from_slice(&SCALING_FACTORS);

    SuperResolutionEngineProperties {
        vendor_id: physical_device.properties().vendor_id,
        engine_version: 0,
        engine_uuid: [0u8; vk::UUID_SIZE],
        engine_name: engine_name(b"NVIDIA DLSS Ray Reconstruction"),
        flags: SuperResolutionEnginePropertyFlags::IS_TEMPORAL
            | SuperResolutionEnginePropertyFlags::SUPPORTS_DYNAMIC_SOURCE_SIZE,
        image_used: SuperResolutionImageUseFlags::SOURCE
            | SuperResolutionImageUseFlags::DESTINATION
            | SuperResolutionImageUseFlags::DEPTH
            | SuperResolutionImageUseFlags::MOTION_VECTORS
            | SuperResolutionImageUseFlags::NORMAL
            | SuperResolutionImageUseFlags::DIFFUSE_ALBEDO
            | SuperResolutionImageUseFlags::SPECULAR_ALBEDO
            | SuperResolutionImageUseFlags::ROUGHNESS
            | SuperResolutionImageUseFlags::SPECULAR_HIT_DISTANCE,
        // DLSS exposes quality via render-scale (the scaling factors above), not
        // an orthogonal quality/perf/power focus; report the single balanced one.
        supported_quality_focuses: SuperResolutionQualityFocusFlags::BALANCED,
        supported_queue_family_indexes: [vk::QUEUE_FAMILY_IGNORED;
            MAX_SUPER_RESOLUTION_QUEUE_FAMILY_COUNT],
        supported_scaling_factor_count: SCALING_FACTORS.len() as u32,
        supported_scaling_factors,
        max_destination_region_size: vk::Extent2D { width: 16384, height: 16384 },
        max_supported_concurrent_session_dispatches: 1,
    }
}

pub(crate) fn engine_supported_image_properties(
    engine: SuperResolutionEngine,
    image_use: SuperResolutionImageUseFlags,
) -> Vec<SuperResolutionImageProperties> {
    assert!(
        engine == SuperResolutionEngine::DLSS_RR,
        "unknown super resolution engine for the DLSS backend"
    );
    use SuperResolutionImageUseFlags as Use;
    use vk::Format;

    let storage = vk::ImageUsageFlags::STORAGE;
    let entries: [(Use, &[Format], vk::ImageUsageFlags); 9] = [
        (Use::SOURCE, COLOR_FORMATS, storage),
        (Use::DESTINATION, COLOR_FORMATS, storage),
        (Use::DEPTH, &[Format::R32_SFLOAT, Format::D32_SFLOAT], storage),
        (Use::MOTION_VECTORS, &[Format::R16G16_SFLOAT], storage),
        (Use::NORMAL, &[Format::R16G16B16A16_SFLOAT], storage),
        (
            Use::DIFFUSE_ALBEDO,
            &[Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB],
            storage,
        ),
        (Use::SPECULAR_ALBEDO, &[Format::R8G8B8A8_UNORM], storage),
        (
            Use::ROUGHNESS,
            &[Format::R8_UNORM, Format::R16_SFLOAT],
            storage,
        ),
        (
            Use::SPECULAR_HIT_DISTANCE,
            &[Format::R16_SFLOAT, Format::R32_SFLOAT],
            storage,
        ),
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

/// NGX manages its own working memory, so a session exposes no Vulkan-visible
/// memory bind points.
pub(crate) fn session_memory_requirements() -> Vec<SuperResolutionSessionMemoryRequirements> {
    Vec::new()
}

/// NGX manages its own descriptors, so a session requires no application
/// descriptor-heap ranges.
pub(crate) fn session_descriptor_heap_ranges() -> SuperResolutionDescriptorHeapRanges {
    SuperResolutionDescriptorHeapRanges {
        resource_heap_size: 0,
        resource_heap_alignment: 0,
        sampler_heap_size: 0,
        sampler_heap_alignment: 0,
    }
}

/// State backing a DLSS-RR [`crate::SuperResolutionSession`]. The NGX feature is
/// created lazily in [`initialize_session`] (NGX feature creation needs a
/// command buffer), so this holds the deferred create params until then.
pub(crate) struct DlssSession {
    create_params: sys::NVSDK_NGX_DLSSD_Create_Params,
    feature: Mutex<Option<NgxFeature>>,
    destination_extent: vk::Extent2D,
    mv_scale_x: f32,
    mv_scale_y: f32,
}

/// Whether a color format carries values beyond the `[0, 1]` range (HDR).
fn is_hdr_format(format: vk::Format) -> bool {
    format == vk::Format::R16G16B16A16_SFLOAT || format == vk::Format::R32G32B32A32_SFLOAT
}

/// Maps a destination/source ratio to the nearest NGX perf-quality value.
fn perf_quality_for(
    destination: vk::Extent2D,
    source: vk::Extent2D,
) -> sys::NVSDK_NGX_PerfQuality_Value {
    use sys::NVSDK_NGX_PerfQuality_Value as Q;
    let ratio = destination.width as f32 / source.width.max(1) as f32;
    if ratio < 1.15 {
        Q::DLAA
    } else if ratio < 1.4 {
        Q::UltraQuality
    } else if ratio < 1.6 {
        Q::MaxQuality
    } else if ratio < 1.85 {
        Q::Balanced
    } else if ratio < 2.5 {
        Q::MaxPerf
    } else {
        Q::UltraPerformance
    }
}

/// Translates a [`SuperResolutionSessionCreateInfo`] into NGX DLSS-RR create
/// params.
fn build_create_params(ci: &SuperResolutionSessionCreateInfo) -> sys::NVSDK_NGX_DLSSD_Create_Params {
    use sys::NVSDK_NGX_DLSS_Feature_Flags as Flag;

    let mut flags = Flag::None;
    if is_hdr_format(ci.source_format)
        && !ci
            .flags
            .contains(SuperResolutionSessionCreateFlags::FORCE_LDR_COLORS)
    {
        flags |= Flag::IsHDR;
    }
    if !ci
        .flags
        .contains(SuperResolutionSessionCreateFlags::MOTION_VECTORS_USE_DESTINATION_DIMENSIONS)
    {
        flags |= Flag::MVLowRes;
    }
    if ci
        .flags
        .contains(SuperResolutionSessionCreateFlags::INVERTED_DEPTH)
    {
        flags |= Flag::DepthInverted;
    }
    if ci
        .flags
        .contains(SuperResolutionSessionCreateFlags::MOTION_VECTORS_USE_JITTER)
    {
        flags |= Flag::MVJittered;
    }
    if ci
        .flags
        .contains(SuperResolutionSessionCreateFlags::USE_AUTO_EXPOSURE)
    {
        flags |= Flag::AutoExposure;
    }

    sys::NVSDK_NGX_DLSSD_Create_Params {
        InDenoiseMode: sys::NVSDK_NGX_DLSS_Denoise_Mode::DLUnified,
        // No separate roughness G-buffer => roughness is packed in normals.w.
        InRoughnessMode: if ci.roughness_format == vk::Format::UNDEFINED {
            sys::NVSDK_NGX_DLSS_Roughness_Mode::Packed
        } else {
            sys::NVSDK_NGX_DLSS_Roughness_Mode::Unpacked
        },
        InUseHWDepth: if ci
            .flags
            .contains(SuperResolutionSessionCreateFlags::LINEAR_DEPTH)
        {
            sys::NVSDK_NGX_DLSS_Depth_Type::Linear
        } else {
            sys::NVSDK_NGX_DLSS_Depth_Type::HW
        },
        InWidth: ci.max_source_region_size.width,
        InHeight: ci.max_source_region_size.height,
        InTargetWidth: ci.destination_region_size.width,
        InTargetHeight: ci.destination_region_size.height,
        InPerfQualityValue: perf_quality_for(
            ci.destination_region_size,
            ci.max_source_region_size,
        ),
        InFeatureCreateFlags: flags,
        InEnableOutputSubrects: ci
            .flags
            .contains(SuperResolutionSessionCreateFlags::ALLOW_SUBRECT_DESTINATION),
    }
}

/// Implements [`crate::SuperResolutionSession::new`] for the DLSS backend.
///
/// Defers NGX feature creation: stores the translated create params and returns
/// a session whose feature is filled in by [`initialize_session`].
pub(crate) fn create_session(
    pipeline_cache: &PipelineCache,
    create_info: &SuperResolutionSessionCreateInfo,
    application_info: &crate::SuperResolutionApplicationInfo<'_>,
) -> VkResult<crate::SuperResolutionSession> {
    assert!(
        create_info.engine == SuperResolutionEngine::DLSS_RR,
        "unknown super resolution engine for the DLSS backend"
    );

    // Availability was confirmed at enumeration; lazily initialize the NGX
    // runtime now that a logical device (required by NGX Init) is available. The
    // application identity is registered with NGX here, at the first Init.
    ensure_runtime(pipeline_cache.device(), application_info)
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;

    Ok(crate::SuperResolutionSession {
        device: pipeline_cache.device().clone(),
        dlss: DlssSession {
            create_params: build_create_params(create_info),
            feature: Mutex::new(None),
            destination_extent: create_info.destination_region_size,
            mv_scale_x: create_info.motion_vector_scale_x,
            mv_scale_y: create_info.motion_vector_scale_y,
        },
    })
}

/// Implements [`crate::SuperResolutionCommandEncoder::initialize_super_resolution_session`].
///
/// Records NGX feature creation into the encoder's command buffer and stores the
/// resulting feature on the session.
pub(crate) fn initialize_session(
    encoder: &mut CommandEncoder,
    session: &crate::SuperResolutionSession,
) {
    let cmd_buffer = encoder.buffer().vk_handle();
    match create_dlssd_feature(cmd_buffer, &session.dlss.create_params) {
        Ok(feature) => {
            *session.dlss.feature.lock().unwrap() = Some(feature);
        }
        Err(e) => {
            tracing::error!(target: "ngx", "DLSS-RR feature creation failed: {e}");
        }
    }
}

/// Creates a transient image view from `image_info`'s create-info and builds an
/// NGX resource over it. The view is destroyed once the command buffer
/// completes (via [`CommandEncoder::retain`]).
///
/// # Safety
/// `image_info.view` must be a valid image-view create-info whose image stays
/// alive until the recorded command buffer finishes.
unsafe fn make_resource(
    encoder: &mut CommandEncoder,
    image_info: &SuperResolutionImageInfo,
    width: u32,
    height: u32,
    read_write: bool,
) -> Resource {
    let device = encoder.device().clone();
    // SAFETY: the caller guarantees a valid create-info; the view lives until
    // the retained guard drops on GPU completion.
    let view = unsafe { device.create_image_view(image_info.view, None) }
        .expect("DLSS: failed to create transient image view");
    encoder.retain(ViewGuard {
        device: device.clone(),
        view,
    });
    Resource::image_view(
        view,
        image_info.view.image,
        image_info.view.subresource_range,
        image_info.view.format,
        width,
        height,
        read_write,
    )
}

/// Destroys a transient image view when the owning command buffer completes.
struct ViewGuard {
    device: Device,
    view: vk::ImageView,
}

impl Drop for ViewGuard {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image_view(self.view, None) };
    }
}

/// Flattens a column-major 4x4 matrix into 16 contiguous floats for NGX.
fn flatten(m: [[f32; 4]; 4]) -> [f32; 16] {
    // `[[f32; 4]; 4]` is 16 contiguous f32 with column order preserved.
    unsafe { core::mem::transmute(m) }
}

/// Implements [`crate::SuperResolutionCommandEncoder::dispatch_super_resolution`].
pub(crate) fn dispatch(
    encoder: &mut CommandEncoder,
    session: &crate::SuperResolutionSession,
    info: &SuperResolutionDispatchInfo,
) {
    let feature_guard = session.dlss.feature.lock().unwrap();
    let Some(feature) = feature_guard.as_ref() else {
        tracing::warn!(target: "ngx", "DLSS-RR dispatch skipped; session not initialized");
        return;
    };

    let src = info.source_size;
    let dst = session.dlss.destination_extent;
    let packed_roughness = session.dlss.create_params.InRoughnessMode
        == sys::NVSDK_NGX_DLSS_Roughness_Mode::Packed;
    let denoise = info.denoise_info;

    // SAFETY: every image create-info originates from the caller's live G-buffer
    // resources, which must outlive this command buffer's GPU execution.
    unsafe {
        let mut color = make_resource(encoder, info.source_image_info, src.width, src.height, false);
        let mut output =
            make_resource(encoder, info.destination_image_info, dst.width, dst.height, true);
        let mut depth = info
            .source_depth_image_info
            .map(|d| make_resource(encoder, d, src.width, src.height, false));
        let mut motion = info
            .motion_info
            .and_then(|m| m.motion_vectors_image_info)
            .map(|mv| make_resource(encoder, mv, src.width, src.height, false));
        let mut diffuse = denoise
            .map(|d| make_resource(encoder, d.diffuse_albedo_image_info, src.width, src.height, false));
        let mut specular = denoise.map(|d| {
            make_resource(encoder, d.specular_albedo_image_info, src.width, src.height, false)
        });
        let mut normals = denoise
            .map(|d| make_resource(encoder, d.normal_image_info, src.width, src.height, false));
        let mut roughness = if packed_roughness {
            None
        } else {
            denoise
                .map(|d| make_resource(encoder, d.roughness_image_info, src.width, src.height, false))
        };
        let mut specular_hit = denoise
            .and_then(|d| d.specular_hit_distance_image_info)
            .map(|r| make_resource(encoder, r, src.width, src.height, false));
        let mut world_to_view = denoise.map(|d| flatten(d.world_to_view_matrix));
        let mut view_to_clip = denoise.map(|d| flatten(d.view_to_clip_matrix));

        let mut eval = sys::NVSDK_NGX_VK_DLSSD_Eval_Params::zeroed();
        eval.pInColor = color.as_mut_ptr();
        eval.pInOutput = output.as_mut_ptr();
        if let Some(d) = depth.as_mut() {
            eval.pInDepth = d.as_mut_ptr();
        }
        if let Some(m) = motion.as_mut() {
            eval.pInMotionVectors = m.as_mut_ptr();
        }
        if let Some(d) = diffuse.as_mut() {
            eval.pInDiffuseAlbedo = d.as_mut_ptr();
        }
        if let Some(s) = specular.as_mut() {
            eval.pInSpecularAlbedo = s.as_mut_ptr();
        }
        if let Some(n) = normals.as_mut() {
            eval.pInNormals = n.as_mut_ptr();
        }
        if let Some(r) = roughness.as_mut() {
            eval.pInRoughness = r.as_mut_ptr();
        }
        if let Some(h) = specular_hit.as_mut() {
            eval.pInSpecularHitDistance = h.as_mut_ptr();
        }
        if let Some(m) = world_to_view.as_mut() {
            eval.pInWorldToViewMatrix = m.as_mut_ptr();
        }
        if let Some(m) = view_to_clip.as_mut() {
            eval.pInViewToClipMatrix = m.as_mut_ptr();
        }

        eval.InRenderSubrectDimensions = sys::NVSDK_NGX_Dimensions {
            Width: src.width,
            Height: src.height,
        };
        eval.InMVScaleX = session.dlss.mv_scale_x;
        eval.InMVScaleY = session.dlss.mv_scale_y;
        if let Some(m) = info.motion_info {
            eval.InJitterOffsetX = m.texel_jitter_x;
            eval.InJitterOffsetY = m.texel_jitter_y;
        }
        eval.InReset = info
            .flags
            .contains(SuperResolutionDispatchFlags::RESET_HISTORY) as c_int;

        let cmd_buffer = encoder.buffer().vk_handle();
        if let Err(e) = evaluate_dlssd(cmd_buffer, feature, &mut eval) {
            tracing::error!(target: "ngx", "DLSS-RR evaluate failed: {e}");
        }
    }
}

/// Implements [`crate::SuperResolutionSession::recommended_jitter_pattern`].
/// NGX exposes no jitter API, so this returns a Halton(2, 3) sequence.
///
/// DLSS-RR guidance: "there is no reason to limit the number of samples; use of
/// many more jitter positions (at least 32) is also highly recommended." So the
/// phase count is the DLSS super-resolution base of `8 * ratio²`, floored at the
/// DLSS-RR minimum of 32.
pub(crate) fn recommended_jitter_pattern(
    destination_region_size: vk::Extent2D,
    source_region_size: vk::Extent2D,
) -> VkResult<Vec<(f32, f32)>> {
    let scale = destination_region_size.width as f32 / source_region_size.width.max(1) as f32;
    let phase_count = ((8.0 * scale * scale) as u32).max(32);
    Ok((1..=phase_count)
        .map(|i| (halton(i, 2) - 0.5, halton(i, 3) - 0.5))
        .collect())
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
