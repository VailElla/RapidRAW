use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, ImageFormat, Luma, imageops};
use jxl_encoder::{ImageMetadata as JxlImageMetadata, LosslessConfig, LossyConfig, PixelLayout};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tauri::Emitter;
use tauri::Manager;

use crate::AppState;
use crate::exif_processing;
use crate::file_management::{
    generate_filename_from_template, parse_virtual_path, read_file_mapped,
};
use crate::formats::is_raw_file;
use crate::image_loader::{
    composite_patches_on_image, load_and_composite, load_base_image_from_bytes,
};
use crate::image_processing::{
    AllAdjustments, Crop, GpuContext, RenderRequest, downscale_f32_image,
    get_all_adjustments_from_json, get_or_init_gpu_context, process_and_get_dynamic_image,
    process_and_get_high_precision_dynamic_image, resolve_tonemapper_override_from_handle,
};
use crate::lut_processing::{
    convert_image_to_cube_lut, generate_identity_lut_image, get_or_load_lut,
};
use crate::mask_generation::{MaskDefinition, generate_mask_bitmap};

use crate::cache_utils::{calculate_full_job_hash, calculate_transform_hash};
use crate::{
    apply_all_transformations, generate_transformed_preview, get_cached_or_generate_mask,
    hydrate_adjustments, load_settings, resolve_warped_image_for_masks,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum ResizeMode {
    LongEdge,
    ShortEdge,
    Width,
    Height,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResizeOptions {
    pub mode: ResizeMode,
    pub value: u32,
    pub dont_enlarge: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    pub jpeg_quality: u8,
    #[serde(default = "default_jxl_bit_depth")]
    pub jxl_bit_depth: u8,
    #[serde(default = "default_jxl_effort")]
    pub jxl_effort: u8,
    pub resize: Option<ResizeOptions>,
    pub keep_metadata: bool,
    #[serde(default)]
    pub preserve_timestamps: bool,
    pub strip_gps: bool,
    pub filename_template: Option<String>,
    pub watermark: Option<WatermarkSettings>,
    #[serde(default)]
    pub export_masks: bool,
    #[serde(default)]
    pub preserve_folders: bool,
}

const fn default_jxl_bit_depth() -> u8 {
    8
}

const fn default_jxl_effort() -> u8 {
    5
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum WatermarkAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WatermarkSettings {
    pub path: String,
    pub anchor: WatermarkAnchor,
    pub scale: f32,
    pub spacing: f32,
    pub opacity: f32,
}

fn apply_watermark(
    base_image: &mut DynamicImage,
    watermark_settings: &WatermarkSettings,
) -> Result<(), String> {
    let watermark_img = image::open(&watermark_settings.path)
        .map_err(|e| format!("Failed to open watermark image: {}", e))?;

    let (base_w, base_h) = base_image.dimensions();
    let base_min_dim = base_w.min(base_h) as f32;

    let watermark_scale_factor =
        (base_min_dim * (watermark_settings.scale / 100.0)) / watermark_img.width().max(1) as f32;
    let new_wm_w = (watermark_img.width() as f32 * watermark_scale_factor).round() as u32;
    let new_wm_h = (watermark_img.height() as f32 * watermark_scale_factor).round() as u32;

    if new_wm_w == 0 || new_wm_h == 0 {
        return Ok(());
    }

    let scaled_watermark =
        watermark_img.resize_exact(new_wm_w, new_wm_h, image::imageops::FilterType::Lanczos3);
    let mut scaled_watermark_rgba = scaled_watermark.to_rgba8();

    let opacity_factor = (watermark_settings.opacity / 100.0).clamp(0.0, 1.0);
    for pixel in scaled_watermark_rgba.pixels_mut() {
        pixel[3] = (pixel[3] as f32 * opacity_factor) as u8;
    }
    let spacing_pixels = (base_min_dim * (watermark_settings.spacing / 100.0)) as i64;
    let (wm_w, wm_h) = scaled_watermark_rgba.dimensions();

    let x = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::CenterLeft | WatermarkAnchor::BottomLeft => {
            spacing_pixels
        }
        WatermarkAnchor::TopCenter | WatermarkAnchor::Center | WatermarkAnchor::BottomCenter => {
            (base_w as i64 - wm_w as i64) / 2
        }
        WatermarkAnchor::TopRight | WatermarkAnchor::CenterRight | WatermarkAnchor::BottomRight => {
            base_w as i64 - wm_w as i64 - spacing_pixels
        }
    };

    let y = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::TopCenter | WatermarkAnchor::TopRight => {
            spacing_pixels
        }
        WatermarkAnchor::CenterLeft | WatermarkAnchor::Center | WatermarkAnchor::CenterRight => {
            (base_h as i64 - wm_h as i64) / 2
        }
        WatermarkAnchor::BottomLeft
        | WatermarkAnchor::BottomCenter
        | WatermarkAnchor::BottomRight => base_h as i64 - wm_h as i64 - spacing_pixels,
    };

    match base_image {
        DynamicImage::ImageRgb32F(base) => {
            for (source_x, source_y, source) in scaled_watermark_rgba.enumerate_pixels() {
                let destination_x = x + source_x as i64;
                let destination_y = y + source_y as i64;
                if destination_x < 0
                    || destination_y < 0
                    || destination_x >= base.width() as i64
                    || destination_y >= base.height() as i64
                {
                    continue;
                }

                let alpha = source[3] as f32 / u8::MAX as f32;
                let inverse_alpha = 1.0 - alpha;
                let destination = base.get_pixel_mut(destination_x as u32, destination_y as u32);
                for channel in 0..3 {
                    let source_value = source[channel] as f32 / u8::MAX as f32;
                    destination[channel] =
                        source_value * alpha + destination[channel] * inverse_alpha;
                }
            }
        }
        DynamicImage::ImageRgba32F(base) => {
            for (source_x, source_y, source) in scaled_watermark_rgba.enumerate_pixels() {
                let destination_x = x + source_x as i64;
                let destination_y = y + source_y as i64;
                if destination_x < 0
                    || destination_y < 0
                    || destination_x >= base.width() as i64
                    || destination_y >= base.height() as i64
                {
                    continue;
                }

                let source_alpha = source[3] as f32 / u8::MAX as f32;
                let destination = base.get_pixel_mut(destination_x as u32, destination_y as u32);
                let destination_alpha = destination[3].clamp(0.0, 1.0);
                let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);

                for channel in 0..3 {
                    let source_value = source[channel] as f32 / u8::MAX as f32;
                    destination[channel] = if output_alpha > 0.0 {
                        (source_value * source_alpha
                            + destination[channel] * destination_alpha * (1.0 - source_alpha))
                            / output_alpha
                    } else {
                        0.0
                    };
                }
                destination[3] = output_alpha;
            }
        }
        _ => {
            let final_watermark = DynamicImage::ImageRgba8(scaled_watermark_rgba);
            image::imageops::overlay(base_image, &final_watermark, x, y);
        }
    }

    Ok(())
}

fn calculate_resize_target(
    current_w: u32,
    current_h: u32,
    resize_opts: &ResizeOptions,
) -> (u32, u32) {
    if resize_opts.dont_enlarge {
        let exceeds = match resize_opts.mode {
            ResizeMode::LongEdge => current_w.max(current_h) > resize_opts.value,
            ResizeMode::ShortEdge => current_w.min(current_h) > resize_opts.value,
            ResizeMode::Width => current_w > resize_opts.value,
            ResizeMode::Height => current_h > resize_opts.value,
        };
        if !exceeds {
            return (current_w, current_h);
        }
    }

    let fix_width = match resize_opts.mode {
        ResizeMode::LongEdge => current_w >= current_h,
        ResizeMode::ShortEdge => current_w <= current_h,
        ResizeMode::Width => true,
        ResizeMode::Height => false,
    };

    let value = resize_opts.value;
    if fix_width {
        let h = (value as f32 * (current_h as f32 / current_w as f32)).round() as u32;
        (value, h)
    } else {
        let w = (value as f32 * (current_w as f32 / current_h as f32)).round() as u32;
        (w, value)
    }
}

fn relative_dir_is_safe(rel_dir: &Path) -> bool {
    rel_dir.components().all(|component| {
        matches!(
            component,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    })
}

#[cfg(windows)]
fn component_matches(left: std::path::Component<'_>, right: std::path::Component<'_>) -> bool {
    left.as_os_str()
        .to_string_lossy()
        .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
}

#[cfg(not(windows))]
fn component_matches(left: std::path::Component<'_>, right: std::path::Component<'_>) -> bool {
    left == right
}

fn strip_prefix_preserving_source_case(source_path: &Path, base_path: &Path) -> Option<PathBuf> {
    let source_components: Vec<_> = source_path.components().collect();
    let base_components: Vec<_> = base_path.components().collect();

    if base_components.len() > source_components.len() {
        return None;
    }

    if !source_components
        .iter()
        .zip(base_components.iter())
        .all(|(source, base)| component_matches(*source, *base))
    {
        return None;
    }

    Some(source_components[base_components.len()..].iter().collect())
}

fn relative_export_dir_for_preserved_folders(
    source_path: &Path,
    base_origin_folders: &[String],
) -> Option<PathBuf> {
    base_origin_folders
        .iter()
        .filter_map(|base| {
            let base_path = Path::new(base);
            strip_prefix_preserving_source_case(source_path, base_path)
                .map(|rel_path| (base_path.components().count(), rel_path))
        })
        .max_by_key(|(component_count, _)| *component_count)
        .and_then(|(_, rel_path)| {
            let rel_dir = rel_path.parent().unwrap_or_else(|| Path::new(""));
            if relative_dir_is_safe(rel_dir) {
                Some(rel_dir.to_path_buf())
            } else {
                None
            }
        })
}

fn apply_export_resize_and_watermark(
    mut image: DynamicImage,
    export_settings: &ExportSettings,
) -> Result<DynamicImage, String> {
    let resize_started = Instant::now();
    let mut resized = false;
    if let Some(resize_opts) = &export_settings.resize {
        let (current_w, current_h) = image.dimensions();
        let (target_w, target_h) = calculate_resize_target(current_w, current_h, resize_opts);

        if target_w != current_w || target_h != current_h {
            image = image.resize(target_w, target_h, imageops::FilterType::Lanczos3);
            resized = true;
        }
    }
    log::info!(
        "Export stage resize (applied={resized}) took {:?}",
        resize_started.elapsed()
    );

    let watermark_started = Instant::now();
    if let Some(watermark_settings) = &export_settings.watermark {
        apply_watermark(&mut image, watermark_settings)?;
    }
    log::info!(
        "Export stage watermark (applied={}) took {:?}",
        export_settings.watermark.is_some(),
        watermark_started.elapsed()
    );
    Ok(image)
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export_pipeline(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    high_precision: bool,
    debug_tag: &str,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage, String> {
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let warped_image = resolve_warped_image_for_masks(state, js_adjustments, &mask_definitions);
    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| {
            generate_mask_bitmap(
                def,
                img_w,
                img_h,
                1.0,
                unscaled_crop_offset,
                warped_image.as_deref(),
            )
        })
        .collect();

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let mut all_adjustments = get_all_adjustments_from_json(js_adjustments, is_raw, tm_override);
    all_adjustments.global.show_clipping = 0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());

    let unique_hash = calculate_full_job_hash(path, js_adjustments);

    let request = RenderRequest {
        adjustments: all_adjustments,
        mask_bitmaps: &mask_bitmaps,
        lut,
        roi: None,
    };
    if high_precision {
        process_and_get_high_precision_dynamic_image(
            context,
            state,
            transformed_image.as_ref(),
            unique_hash,
            request,
            debug_tag,
        )
    } else {
        process_and_get_dynamic_image(
            context,
            state,
            transformed_image.as_ref(),
            unique_hash,
            request,
            debug_tag,
        )
    }
}

fn set_timestamps_from_exif(src: &Path, dst: &Path) {
    let capture_dt = exif_processing::get_creation_date_from_path(src);
    let ft = filetime::FileTime::from_unix_time(
        capture_dt.timestamp(),
        capture_dt.timestamp_subsec_nanos(),
    );
    if let Err(e) = filetime::set_file_times(dst, ft, ft) {
        log::warn!("Could not set timestamps on '{}': {}", dst.display(), e);
    }
}

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const MIN_EXPORT_TASK_BYTES: u64 = 5 * GIB / 2;
const EXPORT_TASK_FIXED_BYTES: u64 = 512 * MIB;
const JXL16_EXPORT_BYTES_PER_PIXEL: u64 = 72;
const STANDARD_EXPORT_BYTES_PER_PIXEL: u64 = 40;
const UNKNOWN_JXL16_PIXELS: u64 = 200_000_000;
const UNKNOWN_STANDARD_PIXELS: u64 = 50_000_000;

fn source_pixel_count(path: &Path) -> Option<u64> {
    if let Ok((width, height)) = image::image_dimensions(path) {
        return u64::from(width).checked_mul(u64::from(height));
    }

    if !is_raw_file(path) {
        return None;
    }

    if let Some(failure) = crate::image_loader::cached_raw_decode_failure_for_path(path) {
        log::warn!(
            "Skipping RAW dimension probe for cached failure '{}': {}",
            path.display(),
            failure
        );
        return None;
    }
    if crate::image_loader::cached_raw_dimension_failure_for_path(path) {
        log::warn!(
            "Skipping cached RAW dimension probe panic for '{}'",
            path.display()
        );
        return None;
    }

    // `dummy=true` asks rawler for the structural RAW image without unpacking
    // its full pixel payload. The file mapping is read-only and short-lived.
    let mapped = read_file_mapped(path).ok()?;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let source = rawler::rawsource::RawSource::new_from_slice(&mapped);
        let decoder = rawler::get_decoder(&source).ok()?;
        let raw = decoder
            .raw_image(&source, &rawler::decoders::RawDecodeParams::default(), true)
            .ok()?;
        u64::try_from(raw.width)
            .ok()?
            .checked_mul(u64::try_from(raw.height).ok()?)
    })) {
        Ok(pixel_count) => pixel_count,
        Err(_) => {
            crate::image_loader::remember_raw_dimension_panic_for_path(path);
            log::warn!("RAW dimension probe panicked for '{}'", path.display());
            None
        }
    }
}

fn largest_export_pixel_count(paths: &[String], jxl16: bool) -> (u64, bool) {
    let unknown_pixels = if jxl16 {
        UNKNOWN_JXL16_PIXELS
    } else {
        UNKNOWN_STANDARD_PIXELS
    };
    let mut used_fallback = false;
    let largest = paths
        .iter()
        .map(|path| {
            let (source_path, _) = parse_virtual_path(path);
            source_pixel_count(&source_path).unwrap_or_else(|| {
                used_fallback = true;
                unknown_pixels
            })
        })
        .max()
        .unwrap_or(unknown_pixels);
    (largest.max(1), used_fallback)
}

fn estimated_export_task_bytes(pixel_count: u64, jxl16: bool) -> u64 {
    let bytes_per_pixel = if jxl16 {
        JXL16_EXPORT_BYTES_PER_PIXEL
    } else {
        STANDARD_EXPORT_BYTES_PER_PIXEL
    };
    EXPORT_TASK_FIXED_BYTES
        .saturating_add(pixel_count.saturating_mul(bytes_per_pixel))
        .max(MIN_EXPORT_TASK_BYTES)
}

fn memory_limited_export_threads(
    path_count: usize,
    available_cores: usize,
    available_memory: u64,
    pixel_count: u64,
    jxl16: bool,
) -> Result<(usize, u64, u64), String> {
    let per_task = estimated_export_task_bytes(pixel_count, jxl16);
    // Preserve headroom for the UI, OS, encoder output and allocator/GPU
    // variance that the byte-per-pixel model cannot observe.
    let memory_budget = available_memory.saturating_mul(85) / 100;
    let memory_limit = memory_budget / per_task;
    if jxl16 && memory_limit == 0 {
        return Err(format!(
            "Insufficient free memory for {:.1} MP 16-bit JXL export: estimated {:.1} GiB task peak, {:.1} GiB safe budget",
            pixel_count as f64 / 1_000_000.0,
            per_task as f64 / GIB as f64,
            memory_budget as f64 / GIB as f64,
        ));
    }

    let threads = path_count
        .max(1)
        .min(available_cores.max(1))
        .min(memory_limit.max(1) as usize)
        .min(16);
    Ok((threads, per_task, memory_budget))
}

fn normalize_jxl_alpha_semantics(image: DynamicImage, source_uses_alpha: bool) -> DynamicImage {
    if source_uses_alpha {
        image
    } else {
        match image {
            DynamicImage::ImageRgba8(buffer) => {
                DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(buffer).to_rgb8())
            }
            DynamicImage::ImageRgba16(buffer) => {
                DynamicImage::ImageRgb16(DynamicImage::ImageRgba16(buffer).to_rgb16())
            }
            DynamicImage::ImageRgba32F(buffer) => {
                DynamicImage::ImageRgb32F(DynamicImage::ImageRgba32F(buffer).to_rgb32f())
            }
            other => other,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SemanticExifValue {
    Byte(Vec<u8>),
    Ascii(Vec<Vec<u8>>),
    Short(Vec<u16>),
    Long(Vec<u32>),
    Rational(Vec<(u32, u32)>),
    SByte(Vec<i8>),
    Undefined(Vec<u8>),
    SShort(Vec<i16>),
    SLong(Vec<i32>),
    SRational(Vec<(i32, i32)>),
    Float(Vec<u32>),
    Double(Vec<u64>),
}

fn semantic_exif_value(value: &exif::Value) -> Result<SemanticExifValue, String> {
    Ok(match value {
        exif::Value::Byte(values) => SemanticExifValue::Byte(values.clone()),
        exif::Value::Ascii(values) => SemanticExifValue::Ascii(values.clone()),
        exif::Value::Short(values) => SemanticExifValue::Short(values.clone()),
        exif::Value::Long(values) => SemanticExifValue::Long(values.clone()),
        exif::Value::Rational(values) => SemanticExifValue::Rational(
            values
                .iter()
                .map(|value| (value.num, value.denom))
                .collect(),
        ),
        exif::Value::SByte(values) => SemanticExifValue::SByte(values.clone()),
        // The second Undefined member is the source-buffer offset. It is a
        // serialization detail, not part of the EXIF value.
        exif::Value::Undefined(values, _) => SemanticExifValue::Undefined(values.clone()),
        exif::Value::SShort(values) => SemanticExifValue::SShort(values.clone()),
        exif::Value::SLong(values) => SemanticExifValue::SLong(values.clone()),
        exif::Value::SRational(values) => SemanticExifValue::SRational(
            values
                .iter()
                .map(|value| (value.num, value.denom))
                .collect(),
        ),
        exif::Value::Float(values) => {
            SemanticExifValue::Float(values.iter().map(|value| value.to_bits()).collect())
        }
        exif::Value::Double(values) => {
            SemanticExifValue::Double(values.iter().map(|value| value.to_bits()).collect())
        }
        exif::Value::Unknown(_, _, _) => {
            return Err("JXL EXIF contains an unsupported unknown value type".to_string());
        }
    })
}

fn semantic_exif_fields(
    exif: &exif::Exif,
) -> Result<BTreeMap<(u16, exif::Tag), Vec<SemanticExifValue>>, String> {
    let mut fields = BTreeMap::new();
    for field in exif.fields() {
        fields
            .entry((field.ifd_num.index(), field.tag))
            .or_insert_with(Vec::new)
            .push(semantic_exif_value(&field.value)?);
    }
    for values in fields.values_mut() {
        values.sort_unstable();
    }
    Ok(fields)
}

fn short_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(&digest[..8])
}

/// Returns the complete TIFF payload from the first Exif box in a JXL
/// container. The Exif box is commonly placed after the codestream. Image
/// decoders are allowed to stop once that codestream is complete, so their
/// auxiliary-box view can contain only the prefix that happened to share the
/// final input buffer. Walking the ISO BMFF boxes by their declared sizes keeps
/// metadata verification independent of that decoder buffering detail.
fn extract_jxl_exif_tiff(image_bytes: &[u8]) -> Result<Option<&[u8]>, String> {
    let mut position = 0usize;

    while position < image_bytes.len() {
        let header = image_bytes
            .get(position..position.saturating_add(8))
            .ok_or_else(|| "JXL container has a truncated box header".to_string())?;
        let size32 = u32::from_be_bytes(header[..4].try_into().unwrap());
        let box_type = &header[4..8];
        let (header_size, box_size) = match size32 {
            0 => (8usize, image_bytes.len() - position),
            1 => {
                let extended = image_bytes
                    .get(position + 8..position + 16)
                    .ok_or_else(|| {
                        "JXL container has a truncated extended box header".to_string()
                    })?;
                let size = u64::from_be_bytes(extended.try_into().unwrap());
                let size = usize::try_from(size)
                    .map_err(|_| "JXL container box is too large".to_string())?;
                (16usize, size)
            }
            size => (8usize, size as usize),
        };

        if box_size < header_size {
            return Err("JXL container has an invalid box size".to_string());
        }
        let box_end = position
            .checked_add(box_size)
            .filter(|end| *end <= image_bytes.len())
            .ok_or_else(|| "JXL container has a truncated box payload".to_string())?;

        if box_type == b"Exif" {
            let payload = &image_bytes[position + header_size..box_end];
            let offset_bytes = payload
                .get(..4)
                .ok_or_else(|| "JXL Exif box is missing its TIFF offset".to_string())?;
            let offset = u32::from_be_bytes(offset_bytes.try_into().unwrap()) as usize;
            let tiff_payload = &payload[4..];
            let tiff = tiff_payload
                .get(offset..)
                .ok_or_else(|| "JXL Exif TIFF offset is invalid".to_string())?;
            return Ok(Some(tiff));
        }

        position = box_end;
        if size32 == 0 {
            break;
        }
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn verify_jxl_export(
    image_bytes: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    samples_per_pixel: u8,
    keep_metadata: bool,
    strip_gps: bool,
    expected_exif_tiff: Option<&[u8]>,
) -> Result<(), String> {
    const CONTAINER_SIGNATURE: [u8; 12] = [
        0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ];
    const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];

    if keep_metadata {
        if !image_bytes.starts_with(&CONTAINER_SIGNATURE) {
            return Err("JXL metadata export is not a valid container".to_string());
        }
    } else if !image_bytes.starts_with(&CODESTREAM_SIGNATURE)
        && !image_bytes.starts_with(&CONTAINER_SIGNATURE)
    {
        return Err("JXL export has an invalid signature".to_string());
    }

    let decoded = jxl_oxide::JxlImage::read_with_defaults(Cursor::new(image_bytes))
        .map_err(|error| format!("Failed to verify JXL header: {error}"))?;
    if decoded.width() != width || decoded.height() != height {
        return Err(format!(
            "JXL dimension verification failed: expected {width}x{height}, got {}x{}",
            decoded.width(),
            decoded.height()
        ));
    }
    let actual_bit_depth = decoded.image_header().metadata.bit_depth.bits_per_sample();
    if actual_bit_depth != u32::from(bit_depth) {
        return Err(format!(
            "JXL bit-depth verification failed: expected {bit_depth}, got {actual_bit_depth}"
        ));
    }
    let pixel_format = decoded.pixel_format();
    let expected_alpha = samples_per_pixel == 4;
    if pixel_format.has_alpha() != expected_alpha {
        return Err(format!(
            "JXL alpha verification failed: expected alpha={expected_alpha}, got alpha={}",
            pixel_format.has_alpha()
        ));
    }
    let actual_samples = pixel_format.channels();
    if actual_samples != usize::from(samples_per_pixel) {
        return Err(format!(
            "JXL channel verification failed: expected {samples_per_pixel}, got {actual_samples}"
        ));
    }
    if decoded.image_header().metadata.orientation != 1 {
        return Err("JXL codestream Orientation is not 1".to_string());
    }

    let exif_tiff = if image_bytes.starts_with(&CONTAINER_SIGNATURE) {
        extract_jxl_exif_tiff(image_bytes)?
    } else {
        None
    };
    match (keep_metadata, exif_tiff) {
        (false, None) => Ok(()),
        (false, Some(_)) => Err("JXL unexpectedly contains an Exif box".to_string()),
        (true, Some(tiff)) => {
            let expected_tiff = expected_exif_tiff
                .ok_or_else(|| "Expected JXL EXIF TIFF payload is missing".to_string())?;
            if tiff != expected_tiff {
                log::warn!(
                    "JXL EXIF payload bytes differ; validating the typed tag contract instead (expected_len={}, actual_len={}, expected_sha256={}, actual_sha256={})",
                    expected_tiff.len(),
                    tiff.len(),
                    short_sha256(expected_tiff),
                    short_sha256(tiff),
                );
            }
            let exif = exif::Reader::new()
                .read_raw(tiff.to_vec())
                .map_err(|error| format!("Failed to reparse JXL EXIF: {error}"))?;
            let expected_exif = exif::Reader::new()
                .read_raw(expected_tiff.to_vec())
                .map_err(|error| format!("Failed to reparse expected JXL EXIF: {error}"))?;
            let actual_fields = semantic_exif_fields(&exif)?;
            let expected_fields = semantic_exif_fields(&expected_exif)?;
            if actual_fields != expected_fields {
                return Err(format!(
                    "JXL EXIF semantic verification failed: expected {} typed fields, got {}",
                    expected_fields.len(),
                    actual_fields.len()
                ));
            }

            let uint_value = |tag| {
                exif.get_field(tag, exif::In::PRIMARY)
                    .and_then(|field| field.value.get_uint(0))
            };
            if uint_value(exif::Tag::ImageWidth) != Some(width)
                || uint_value(exif::Tag::ImageLength) != Some(height)
            {
                return Err("JXL EXIF dimensions do not match the rendered image".to_string());
            }
            if uint_value(exif::Tag::SamplesPerPixel) != Some(u32::from(samples_per_pixel)) {
                return Err("JXL EXIF SamplesPerPixel verification failed".to_string());
            }
            let bits = exif
                .get_field(exif::Tag::BitsPerSample, exif::In::PRIMARY)
                .ok_or_else(|| "JXL EXIF BitsPerSample is missing".to_string())?;
            if (0..usize::from(samples_per_pixel))
                .any(|index| bits.value.get_uint(index) != Some(u32::from(bit_depth)))
            {
                return Err("JXL EXIF BitsPerSample verification failed".to_string());
            }
            if uint_value(exif::Tag::Orientation) != Some(1)
                || uint_value(exif::Tag::ColorSpace) != Some(1)
            {
                return Err("JXL EXIF orientation or color space verification failed".to_string());
            }
            let software = exif
                .get_field(exif::Tag::Software, exif::In::PRIMARY)
                .map(|field| field.display_value().to_string())
                .unwrap_or_default();
            if !software.contains("RapidRAW") {
                return Err("JXL EXIF Software verification failed".to_string());
            }

            if strip_gps
                && exif
                    .fields()
                    .any(|field| field.tag.context() == exif::Context::Gps)
            {
                return Err("JXL EXIF GPS stripping verification failed".to_string());
            }
            Ok(())
        }
        (true, None) => Err("JXL Exif box is missing or incomplete".to_string()),
    }
}

fn ensure_export_not_cancelled(cancellation_token: &AtomicBool) -> Result<(), String> {
    if cancellation_token.load(Ordering::SeqCst) {
        Err("Export cancelled".to_string())
    } else {
        Ok(())
    }
}

fn save_image_with_metadata(
    image: &DynamicImage,
    output_path: &std::path::Path,
    source_path_str: &str,
    source_sidecar_path: Option<&Path>,
    export_settings: &ExportSettings,
) -> Result<(), String> {
    let total_started = Instant::now();
    let extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (width, height) = image.dimensions();
    let samples_per_pixel = if image_uses_alpha_channel(image) {
        4
    } else {
        3
    };
    let metadata_started = Instant::now();
    let jxl_exif_tiff = if extension == "jxl" && export_settings.keep_metadata {
        let result = exif_processing::build_jxl_exif_tiff(
            source_path_str,
            source_sidecar_path,
            export_settings.strip_gps,
            exif_processing::RenderedImageMetadata {
                width,
                height,
                bits_per_sample: export_settings.jxl_bit_depth.into(),
                samples_per_pixel: samples_per_pixel.into(),
            },
        );
        log::info!(
            "JXL export stage metadata-build for '{}' took {:?}",
            output_path.display(),
            metadata_started.elapsed()
        );
        Some(result?)
    } else {
        None
    };

    let encode_started = Instant::now();
    let encoded = encode_image_to_bytes(
        image,
        &extension,
        export_settings.jpeg_quality,
        export_settings.jxl_bit_depth,
        export_settings.jxl_effort,
        jxl_exif_tiff.as_deref(),
    );
    if extension == "jxl" {
        log::info!(
            "JXL export stage encode-total for '{}' took {:?}",
            output_path.display(),
            encode_started.elapsed()
        );
    }
    let mut image_bytes = encoded?;

    if extension == "jxl" {
        let verify_started = Instant::now();
        let verification = verify_jxl_export(
            &image_bytes,
            width,
            height,
            export_settings.jxl_bit_depth,
            samples_per_pixel,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            jxl_exif_tiff.as_deref(),
        );
        log::info!(
            "JXL export stage verify for '{}' took {:?}",
            output_path.display(),
            verify_started.elapsed()
        );
        verification?;
    } else {
        exif_processing::write_image_with_metadata(
            &mut image_bytes,
            source_path_str,
            &extension,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            None,
            None,
        )?;
    }

    let write_started = Instant::now();
    let encoded_len = image_bytes.len();
    #[cfg(target_os = "android")]
    {
        let file_name = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "Missing Android export file name".to_string())?;
        crate::android_integration::save_image_bytes_to_android_gallery(
            file_name,
            mime_type_for_extension(&extension),
            &image_bytes,
        )?;
    }

    #[cfg(not(target_os = "android"))]
    fs::write(output_path, &image_bytes).map_err(|e| e.to_string())?;

    if extension == "jxl" {
        log::info!(
            "JXL export stage write for '{}' ({} bytes) took {:?}; total post-render {:?}",
            output_path.display(),
            encoded_len,
            write_started.elapsed(),
            total_started.elapsed()
        );
    }

    Ok(())
}

#[cfg(target_os = "android")]
pub fn mime_type_for_extension(extension: &str) -> &'static str {
    match extension {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        "jxl" => "image/jxl",
        _ => "application/octet-stream",
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    jxl_export: bool,
    high_precision: bool,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage, String> {
    let processed_image = process_image_for_export_pipeline(
        path,
        base_image,
        js_adjustments,
        context,
        state,
        is_raw,
        high_precision,
        "process_image_for_export",
        app_handle,
    )?;
    let processed_image = if jxl_export {
        normalize_jxl_alpha_semantics(processed_image, !is_raw && base_image.color().has_alpha())
    } else {
        processed_image
    };

    apply_export_resize_and_watermark(processed_image, export_settings)
}

fn build_single_mask_adjustments(all: &AllAdjustments, mask_index: usize) -> AllAdjustments {
    let mut single = AllAdjustments {
        global: all.global,
        mask_adjustments: all.mask_adjustments,
        mask_count: 1,
        tile_offset_x: all.tile_offset_x,
        tile_offset_y: all.tile_offset_y,
        mask_atlas_cols: all.mask_atlas_cols,
    };
    single.mask_adjustments[0] = all.mask_adjustments[mask_index];
    for i in 1..single.mask_adjustments.len() {
        single.mask_adjustments[i] = Default::default();
    }
    single
}

fn encode_grayscale_to_png(bitmap: &GrayImage) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    bitmap
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

fn image_uses_alpha_channel(image: &DynamicImage) -> bool {
    image.color().has_alpha()
}

/// Quantizes an sRGB unit sample for JXL integer input.
///
/// Finite values are clamped to `[0, 1]`, scaled to the full 16-bit range,
/// and rounded to the nearest integer. Non-finite render output fails export
/// rather than silently manufacturing a pixel value.
#[inline]
fn quantize_unit_f32_to_u16(value: f32) -> Result<u16, String> {
    if !value.is_finite() {
        return Err("Cannot export JXL: final image contains a non-finite sample".to_string());
    }
    Ok((value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16)
}

fn quantize_image_to_jxl_u16(image: &DynamicImage, has_alpha: bool) -> Result<Vec<u16>, String> {
    match (image, has_alpha) {
        (DynamicImage::ImageRgba32F(buffer), true) => buffer
            .as_raw()
            .par_iter()
            .copied()
            .map(quantize_unit_f32_to_u16)
            .collect(),
        (DynamicImage::ImageRgba32F(buffer), false) => {
            let source = buffer.as_raw();
            let mut quantized = vec![0u16; source.len() / 4 * 3];
            quantized
                .par_chunks_mut(3)
                .zip(source.par_chunks_exact(4))
                .try_for_each(|(output, input)| {
                    output[0] = quantize_unit_f32_to_u16(input[0])?;
                    output[1] = quantize_unit_f32_to_u16(input[1])?;
                    output[2] = quantize_unit_f32_to_u16(input[2])?;
                    Ok::<(), String>(())
                })?;
            Ok(quantized)
        }
        (DynamicImage::ImageRgb32F(buffer), false) => buffer
            .as_raw()
            .par_iter()
            .copied()
            .map(quantize_unit_f32_to_u16)
            .collect(),
        (_, true) => Ok(image.to_rgba16().into_raw()),
        (_, false) => Ok(image.to_rgb16().into_raw()),
    }
}

fn lossless_jxl_config(jxl_effort: u8) -> LosslessConfig {
    // A single large export can use the ambient Rayon pool's CPU cores;
    // concurrent exports share that bounded pool instead of each creating a
    // dedicated pool and oversubscribing the machine.
    LosslessConfig::new()
        .with_effort(jxl_effort)
        .with_threads(0)
}

fn encode_jxl_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    jpeg_quality: u8,
    jxl_effort: u8,
    jxl_exif_tiff: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let encode_started = Instant::now();
    if !matches!(jxl_effort, 4 | 5 | 7) {
        return Err(format!(
            "Unsupported JXL effort {jxl_effort}; expected 4, 5, or 7"
        ));
    }

    let (mode, effort, distance, result) = if jpeg_quality == 100 {
        let config = lossless_jxl_config(jxl_effort);
        let result = if let Some(exif_tiff) = jxl_exif_tiff {
            let metadata = JxlImageMetadata::new().with_exif(exif_tiff);
            config
                .encode_request(width, height, layout)
                .with_metadata(&metadata)
                .encode(pixels)
                .map_err(|e| format!("Failed to encode lossless JXL: {e}"))
        } else {
            config
                .encode(pixels, width, height, layout)
                .map_err(|e| format!("Failed to encode lossless JXL: {e}"))
        };
        ("lossless", jxl_effort, None, result)
    } else {
        // Preserve RapidRAW's existing quality-slider mapping for lossy JXL.
        let distance = ((100.0 - jpeg_quality as f32) / 10.0).max(0.01);
        // `threads=0` uses the shared ambient Rayon pool. Batch exports then
        // share one bounded pool instead of creating N dedicated pools and
        // oversubscribing the machine.
        let config = LossyConfig::new(distance)
            .with_effort(jxl_effort)
            .with_threads(0);
        let result = if let Some(exif_tiff) = jxl_exif_tiff {
            let metadata = JxlImageMetadata::new().with_exif(exif_tiff);
            config
                .encode_request(width, height, layout)
                .with_metadata(&metadata)
                .encode(pixels)
                .map_err(|e| format!("Failed to encode lossy JXL: {e}"))
        } else {
            config
                .encode(pixels, width, height, layout)
                .map_err(|e| format!("Failed to encode lossy JXL: {e}"))
        };
        ("lossy", jxl_effort, Some(distance), result)
    };

    match &result {
        Ok(bytes) => log::info!(
            "JXL codec encode {width}x{height} mode={mode} effort={effort} distance={distance:?} shared_parallel_threads={} output_bytes={} took {:?}",
            rayon::current_num_threads(),
            bytes.len(),
            encode_started.elapsed()
        ),
        Err(error) => log::warn!(
            "JXL codec encode {width}x{height} mode={mode} effort={effort} distance={distance:?} failed after {:?}: {error}",
            encode_started.elapsed()
        ),
    }
    result
}

fn encode_image_to_bytes(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
    jxl_bit_depth: u8,
    jxl_effort: u8,
    jxl_exif_tiff: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let mut image_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut image_bytes);

    match output_format.to_lowercase().as_str() {
        "jxl" => {
            let (width, height) = image.dimensions();
            let has_alpha = image_uses_alpha_channel(image);

            let jxl_data = match (jxl_bit_depth, has_alpha) {
                (8, true) => {
                    let conversion_started = Instant::now();
                    let converted;
                    let rgba = if let DynamicImage::ImageRgba8(buffer) = image {
                        buffer
                    } else {
                        converted = image.to_rgba8();
                        &converted
                    };
                    log::info!(
                        "JXL export stage pixel-conversion depth=8 channels=4 took {:?}",
                        conversion_started.elapsed()
                    );
                    encode_jxl_pixels(
                        rgba.as_raw(),
                        width,
                        height,
                        PixelLayout::Rgba8,
                        jpeg_quality,
                        jxl_effort,
                        jxl_exif_tiff,
                    )?
                }
                (8, false) => {
                    let conversion_started = Instant::now();
                    let converted;
                    let rgb = if let DynamicImage::ImageRgb8(buffer) = image {
                        buffer
                    } else {
                        converted = image.to_rgb8();
                        &converted
                    };
                    log::info!(
                        "JXL export stage pixel-conversion depth=8 channels=3 took {:?}",
                        conversion_started.elapsed()
                    );
                    encode_jxl_pixels(
                        rgb.as_raw(),
                        width,
                        height,
                        PixelLayout::Rgb8,
                        jpeg_quality,
                        jxl_effort,
                        jxl_exif_tiff,
                    )?
                }
                (16, has_alpha) => {
                    // jxl-encoder's RGB16/RGBA16 layouts explicitly consume
                    // native-endian u16 samples. Casting this u16 allocation
                    // therefore supplies the required host-native byte order
                    // without an additional full-image buffer.
                    let layout = if has_alpha {
                        PixelLayout::Rgba16
                    } else {
                        PixelLayout::Rgb16
                    };
                    let native_samples = match (image, has_alpha) {
                        (DynamicImage::ImageRgba16(buffer), true) => Some(buffer.as_raw()),
                        (DynamicImage::ImageRgb16(buffer), false) => Some(buffer.as_raw()),
                        _ => None,
                    };
                    if let Some(samples) = native_samples {
                        encode_jxl_pixels(
                            bytemuck::cast_slice(samples),
                            width,
                            height,
                            layout,
                            jpeg_quality,
                            jxl_effort,
                            jxl_exif_tiff,
                        )?
                    } else {
                        let quantize_started = Instant::now();
                        let samples = quantize_image_to_jxl_u16(image, has_alpha)?;
                        log::info!(
                            "JXL export stage quantize depth=16 channels={} parallel=true took {:?}",
                            if has_alpha { 4 } else { 3 },
                            quantize_started.elapsed()
                        );
                        encode_jxl_pixels(
                            bytemuck::cast_slice(&samples),
                            width,
                            height,
                            layout,
                            jpeg_quality,
                            jxl_effort,
                            jxl_exif_tiff,
                        )?
                    }
                }
                (other, _) => {
                    return Err(format!(
                        "Unsupported JXL bit depth {other}; expected 8 or 16"
                    ));
                }
            };

            return Ok(jxl_data);
        }
        "webp" => {
            let encoder = webp::Encoder::from_image(image)
                .map_err(|_| "Failed to create WebP encoder".to_string())?;
            let webp_mem = encoder.encode(jpeg_quality as f32);
            return Ok(webp_mem.to_vec());
        }
        "jpg" | "jpeg" => {
            let rgb_image = image.to_rgb8();
            let encoder = JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
            rgb_image
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "png" => {
            let image_to_encode = if image.as_rgb32f().is_some() {
                DynamicImage::ImageRgb16(image.to_rgb16())
            } else {
                image.clone()
            };

            image_to_encode
                .write_to(&mut cursor, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
        }
        "tiff" => {
            DynamicImage::ImageRgb16(image.to_rgb16())
                .write_to(&mut cursor, image::ImageFormat::Tiff)
                .map_err(|e| e.to_string())?;
        }
        "avif" => {
            image
                .write_to(&mut cursor, image::ImageFormat::Avif)
                .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("Unsupported file format: {}", output_format)),
    };
    Ok(image_bytes)
}

#[allow(clippy::too_many_arguments)]
fn export_masks_for_image(
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    output_path_obj: &std::path::Path,
    source_path_str: &str,
    source_sidecar_path: &Path,
    context: &Arc<GpuContext>,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
) -> Result<(), String> {
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
    let (img_w, img_h) = transformed_image.dimensions();
    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let warped_image = resolve_warped_image_for_masks(state, js_adjustments, &mask_definitions);
    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| {
            generate_mask_bitmap(
                def,
                img_w,
                img_h,
                1.0,
                unscaled_crop_offset,
                warped_image.as_deref(),
            )
        })
        .collect();

    if !mask_bitmaps.is_empty() {
        let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
        let all_adjustments = get_all_adjustments_from_json(js_adjustments, is_raw, tm_override);
        let lut_path = js_adjustments["lutPath"].as_str();
        let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());
        let unique_hash = calculate_full_job_hash(source_path_str, js_adjustments);
        let output_dir = output_path_obj.parent().unwrap_or(output_path_obj);
        let stem = output_path_obj
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("export");
        let extension = output_path_obj
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("jpg");

        for (i, _) in mask_bitmaps.iter().enumerate() {
            let single_adjustments = build_single_mask_adjustments(&all_adjustments, i);
            let full_white_mask = ImageBuffer::from_fn(img_w, img_h, |_, _| Luma([255u8]));
            let single_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = vec![full_white_mask];

            let request = RenderRequest {
                adjustments: single_adjustments,
                mask_bitmaps: &single_bitmaps,
                lut: lut.clone(),
                roi: None,
            };
            let processed =
                if extension.eq_ignore_ascii_case("jxl") && export_settings.jxl_bit_depth == 16 {
                    process_and_get_high_precision_dynamic_image(
                        context,
                        state,
                        transformed_image.as_ref(),
                        unique_hash,
                        request,
                        "export_mask_image",
                    )?
                } else {
                    process_and_get_dynamic_image(
                        context,
                        state,
                        transformed_image.as_ref(),
                        unique_hash,
                        request,
                        "export_mask_image",
                    )?
                };
            let processed = if extension.eq_ignore_ascii_case("jxl") {
                normalize_jxl_alpha_semantics(processed, !is_raw && base_image.color().has_alpha())
            } else {
                processed
            };

            let with_options = apply_export_resize_and_watermark(processed, export_settings)?;
            let (out_w, out_h) = with_options.dimensions();

            let alpha_resized = imageops::resize(
                &mask_bitmaps[i],
                out_w,
                out_h,
                imageops::FilterType::Lanczos3,
            );

            let mask_image_path =
                output_dir.join(format!("{}_mask_{}_image.{}", stem, i, extension));
            let mask_alpha_path = output_dir.join(format!("{}_mask_{}_alpha.png", stem, i));

            save_image_with_metadata(
                &with_options,
                &mask_image_path,
                source_path_str,
                Some(source_sidecar_path),
                export_settings,
            )?;

            if export_settings.preserve_timestamps {
                set_timestamps_from_exif(Path::new(source_path_str), &mask_image_path);
            }

            let alpha_bytes = encode_grayscale_to_png(&alpha_resized)?;
            #[cfg(target_os = "android")]
            {
                let file_name = mask_alpha_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| "Missing Android mask export file name".to_string())?;
                crate::android_integration::save_image_bytes_to_android_gallery(
                    file_name,
                    "image/png",
                    &alpha_bytes,
                )?;
            }

            #[cfg(not(target_os = "android"))]
            fs::write(&mask_alpha_path, alpha_bytes).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn export_adjustments_as_lut(
    js_adjustments: &Value,
    source_path_str: &str,
    context: &Arc<GpuContext>,
    state: &tauri::State<AppState>,
    app_handle: &tauri::AppHandle,
) -> Result<Vec<u8>, String> {
    let lut_size = 33;
    let identity_image = generate_identity_lut_image(lut_size);

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, false);
    let mut all_adjustments = get_all_adjustments_from_json(js_adjustments, false, tm_override);

    all_adjustments.global.show_clipping = 0;
    all_adjustments.global.vignette_amount = 0.0;
    all_adjustments.global.grain_amount = 0.0;
    all_adjustments.global.sharpness = 0.0;
    all_adjustments.global.clarity = 0.0;
    all_adjustments.global.dehaze = 0.0;
    all_adjustments.global.structure = 0.0;
    all_adjustments.global.centré = 0.0;
    all_adjustments.global.glow_amount = 0.0;
    all_adjustments.global.halation_amount = 0.0;
    all_adjustments.global.flare_amount = 0.0;
    all_adjustments.global.luma_noise_reduction = 0.0;
    all_adjustments.global.color_noise_reduction = 0.0;
    all_adjustments.global.chromatic_aberration_red_cyan = 0.0;
    all_adjustments.global.chromatic_aberration_blue_yellow = 0.0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());
    let unique_hash = calculate_full_job_hash(source_path_str, js_adjustments);

    let processed_lut = process_and_get_dynamic_image(
        context,
        state,
        &identity_image,
        unique_hash,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &[],
            lut,
            roi: None,
        },
        "export_lut",
    )?;

    convert_image_to_cube_lut(&processed_lut, lut_size)
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn export_images(
    paths: Vec<String>,
    output_folder_or_file: String,
    is_explicit_file_path: bool,
    base_origin_folders: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    if state.export_task_handle.lock().unwrap().is_some() {
        return Err("An export is already in progress.".to_string());
    }
    state
        .export_cancellation_token
        .store(false, Ordering::SeqCst);

    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let context = Arc::new(context);
    let progress_counter = Arc::new(AtomicUsize::new(0));
    let cancellation_token = Arc::clone(&state.export_cancellation_token);

    let available_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut sys = sysinfo::System::new();
    sys.refresh_memory();

    let available_memory = sys.available_memory();
    let available_ram_gb = available_memory as f64 / GIB as f64;
    let jxl16 = output_format.eq_ignore_ascii_case("jxl") && export_settings.jxl_bit_depth == 16;
    let (largest_pixels, used_dimension_fallback) = largest_export_pixel_count(&paths, jxl16);
    let (num_threads, estimated_task_bytes, memory_budget) = memory_limited_export_threads(
        paths.len(),
        available_cores,
        available_memory,
        largest_pixels,
        jxl16,
    )?;

    log::info!(
        "Batch Export: {} cores, {:.1} GiB free, {:.1} MP max input, {:.1} GiB/task estimate, {:.1} GiB safe budget, dimension_fallback={} -> {} threads",
        available_cores,
        available_ram_gb,
        largest_pixels as f64 / 1_000_000.0,
        estimated_task_bytes as f64 / GIB as f64,
        memory_budget as f64 / GIB as f64,
        used_dimension_fallback,
        num_threads
    );

    let task = tokio::spawn(async move {
        let output_folder_path = std::path::Path::new(&output_folder_or_file);
        let total_paths = paths.len();
        let settings = load_settings(app_handle.clone()).unwrap_or_default();

        let mut base_path_counts: HashMap<String, usize> = HashMap::new();
        let mut export_items = Vec::with_capacity(total_paths);

        for (i, path_str) in paths.into_iter().enumerate() {
            let (source_path, _) = parse_virtual_path(&path_str);
            let source_str = source_path.to_string_lossy().to_string();
            let count = base_path_counts.entry(source_str.clone()).or_insert(0);
            *count += 1;

            let mut explicit_vc = None;
            if let Some(idx) = path_str.rfind("vc=") {
                let id_str = path_str[idx + 3..].split('&').next().unwrap_or("");
                if let Ok(id) = id_str.parse::<u32>() {
                    explicit_vc = Some(id);
                }
            }
            if explicit_vc.is_none() {
                let lower = path_str.to_lowercase();
                if let Some(idx) = lower.rfind("_vc") {
                    let id_str: String = lower[idx + 3..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(id) = id_str.parse::<u32>() {
                        explicit_vc = Some(id);
                    }
                }
            }
            export_items.push((i, path_str, *count, explicit_vc));
        }

        let semaphore = Arc::new(tokio::sync::Semaphore::new(num_threads));
        let mut join_handles = Vec::new();

        for (global_index, image_path_str, appearance_count, explicit_vc) in export_items {
            if cancellation_token.load(Ordering::SeqCst) {
                break;
            }
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            if cancellation_token.load(Ordering::SeqCst) {
                drop(permit);
                break;
            }

            let app_handle_clone = app_handle.clone();
            let context_clone = Arc::clone(&context);
            let progress_counter_clone = Arc::clone(&progress_counter);
            let output_folder_path = output_folder_path.to_path_buf();
            let base_origin_folders = base_origin_folders.clone();
            let export_settings = export_settings.clone();
            let output_format = output_format.clone();
            let current_edit_path = current_edit_path.clone();
            let current_edit_adjustments = current_edit_adjustments.clone();
            let settings = settings.clone();
            let cancellation_token_clone = Arc::clone(&cancellation_token);

            let handle = tokio::task::spawn_blocking(move || {
                ensure_export_not_cancelled(&cancellation_token_clone)?;

                let state = app_handle_clone.state::<AppState>();
                let (source_path, sidecar_path) = parse_virtual_path(&image_path_str);
                let source_path_str = source_path.to_string_lossy().to_string();
                let is_current_edit = Some(&source_path_str) == current_edit_path.as_ref();

                let mut js_adjustments = match (is_current_edit, current_edit_adjustments) {
                    (true, Some(adjustments)) => adjustments,
                    _ => {
                        let metadata = crate::exif_processing::load_sidecar(&sidecar_path);
                        metadata.adjustments
                    }
                };

                hydrate_adjustments(&state, &mut js_adjustments);
                let is_raw = is_raw_file(&source_path_str);
                let original_path = std::path::Path::new(&source_path_str);
                let file_date = exif_processing::get_creation_date_from_path(original_path);

                let filename_template = export_settings
                    .filename_template
                    .as_deref()
                    .unwrap_or("{original_filename}_edited");

                let mut new_stem = generate_filename_from_template(
                    filename_template,
                    original_path,
                    global_index + 1,
                    total_paths,
                    &file_date,
                );

                if let Some(vc_id) = explicit_vc {
                    new_stem = format!("{}_VC{:02}", new_stem, vc_id);
                } else if appearance_count > 1 {
                    new_stem = format!("{}_VC{:02}", new_stem, appearance_count - 1);
                }

                let new_filename = format!("{}.{}", new_stem, output_format);
                let output_path = if is_explicit_file_path && total_paths == 1 {
                    output_folder_path
                } else if export_settings.preserve_folders {
                    if let Some(rel_dir) = relative_export_dir_for_preserved_folders(
                        source_path.as_path(),
                        &base_origin_folders,
                    ) {
                        let full_dir = output_folder_path.join(rel_dir);
                        if let Err(e) = std::fs::create_dir_all(&full_dir) {
                            log::warn!("Failed to create export subdirectory: {}", e);
                        }
                        full_dir.join(&new_filename)
                    } else {
                        output_folder_path.join(&new_filename)
                    }
                } else {
                    output_folder_path.join(&new_filename)
                };

                let extension = output_format.to_lowercase();

                let result: Result<(), String> = (|| {
                    if extension == "cube" {
                        let cube_bytes = export_adjustments_as_lut(
                            &js_adjustments,
                            &source_path_str,
                            &context_clone,
                            &state,
                            &app_handle_clone,
                        )?;
                        #[cfg(target_os = "android")]
                        {
                            let file_name = output_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .ok_or_else(|| "Missing Android LUT file name".to_string())?;
                            crate::android_integration::save_file_bytes_to_android_downloads(
                                file_name,
                                "application/octet-stream",
                                &cube_bytes,
                            )?;
                        }
                        #[cfg(not(target_os = "android"))]
                        fs::write(&output_path, cube_bytes).map_err(|e| e.to_string())?;
                        return Ok(());
                    }

                    let base_image = if is_current_edit {
                        match crate::get_original_image(&state) {
                            Ok((orig_data_arc, _)) => {
                                composite_patches_on_image(&orig_data_arc, &js_adjustments)
                                    .map_err(|e| format!("Failed to composite AI patches: {}", e))?
                            }
                            Err(_) => {
                                let bytes =
                                    fs::read(&source_path_str).map_err(|e| e.to_string())?;
                                load_and_composite(
                                    &bytes,
                                    &source_path_str,
                                    &js_adjustments,
                                    false,
                                    &settings,
                                    None,
                                )
                                .map_err(|e| format!("Failed to load fallback image: {}", e))?
                            }
                        }
                    } else {
                        match read_file_mapped(Path::new(&source_path_str)) {
                            Ok(mmap) => load_and_composite(
                                &mmap,
                                &source_path_str,
                                &js_adjustments,
                                false,
                                &settings,
                                None,
                            )
                            .map_err(|e| format!("Failed to load from mmap: {}", e))?,
                            Err(_) => {
                                let bytes =
                                    fs::read(&source_path_str).map_err(|e| e.to_string())?;
                                load_and_composite(
                                    &bytes,
                                    &source_path_str,
                                    &js_adjustments,
                                    false,
                                    &settings,
                                    None,
                                )
                                .map_err(|e| format!("Failed to load from bytes: {}", e))?
                            }
                        }
                    };
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    let mut main_export_adjustments = js_adjustments.clone();
                    if export_settings.export_masks
                        && let Some(obj) = main_export_adjustments.as_object_mut()
                    {
                        obj.insert("masks".to_string(), serde_json::json!([]));
                    }

                    let final_image = process_image_for_export(
                        &source_path_str,
                        &base_image,
                        &main_export_adjustments,
                        &export_settings,
                        &context_clone,
                        &state,
                        is_raw,
                        extension == "jxl",
                        extension == "jxl" && export_settings.jxl_bit_depth == 16,
                        &app_handle_clone,
                    )?;
                    ensure_export_not_cancelled(&cancellation_token_clone)?;
                    save_image_with_metadata(
                        &final_image,
                        &output_path,
                        &source_path_str,
                        Some(&sidecar_path),
                        &export_settings,
                    )?;
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    if export_settings.preserve_timestamps {
                        set_timestamps_from_exif(Path::new(&source_path_str), &output_path);
                    }

                    if export_settings.export_masks {
                        export_masks_for_image(
                            &base_image,
                            &js_adjustments,
                            &export_settings,
                            &output_path,
                            &source_path_str,
                            &sidecar_path,
                            &context_clone,
                            &state,
                            is_raw,
                            &app_handle_clone,
                        )?;
                    }

                    Ok(())
                })();

                if !cancellation_token_clone.load(Ordering::SeqCst) {
                    let current_progress =
                        progress_counter_clone.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = app_handle_clone.emit(
                        "batch-export-progress",
                        serde_json::json!({
                            "current": current_progress,
                            "total": total_paths,
                            "path": &image_path_str
                        }),
                    );
                }

                drop(permit);
                if cancellation_token_clone.load(Ordering::SeqCst) {
                    Err("Export cancelled".to_string())
                } else {
                    result
                }
            });

            join_handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in join_handles {
            match handle.await {
                Ok(res) => results.push(res),
                Err(e) => results.push(Err(format!("Thread crashed: {}", e))),
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        if cancellation_token.load(Ordering::SeqCst) {
            log::info!("Batch export cancelled and worker cleanup completed");
        } else {
            let mut error_count = 0;
            for result in results {
                if let Err(e) = result {
                    error_count += 1;
                    log::error!("Export error: {}", e);
                    if total_paths == 1 {
                        let _ = app_handle.emit("export-error", e);
                    }
                }
            }

            if error_count > 0 && total_paths > 1 {
                let _ = app_handle.emit(
                    "export-complete-with-errors",
                    serde_json::json!({ "errors": error_count, "total": total_paths }),
                );
            } else if error_count == 0 {
                let _ = app_handle.emit(
                    "batch-export-progress",
                    serde_json::json!({ "current": total_paths, "total": total_paths, "path": "" }),
                );
                let _ = app_handle.emit("export-complete", ());
            }
        }

        *app_handle
            .state::<AppState>()
            .export_task_handle
            .lock()
            .unwrap() = None;
    });

    *state.export_task_handle.lock().unwrap() = Some(task);
    Ok(())
}

#[tauri::command]
pub fn cancel_export(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let task_is_active = state.export_task_handle.lock().unwrap().is_some();
    state
        .export_cancellation_token
        .store(true, Ordering::SeqCst);
    let _ = app_handle.emit("export-cancelled", ());
    log::info!(
        "Export cancellation requested (active_task={task_is_active}); workers will stop at the next checkpoint"
    );
    Ok(())
}

#[tauri::command]
pub async fn estimate_export_sizes(
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<usize, String> {
    if output_format.to_lowercase() == "cube" {
        return Ok(1_050_000 * paths.len());
    }

    if paths.is_empty() {
        return Ok(0);
    }

    let first_path = &paths[0];
    let (source_path, sidecar_path) = parse_virtual_path(first_path);
    let source_path_str = source_path.to_string_lossy().to_string();

    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let is_current_edit = Some(&source_path_str) == current_edit_path.as_ref();
    let is_raw = is_raw_file(&source_path_str);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let single_image_extrapolated_size: usize = if is_current_edit
        && current_edit_adjustments.is_some()
    {
        let loaded_image = state
            .original_image
            .lock()
            .unwrap()
            .clone()
            .ok_or("No original image loaded")?;
        let mut adjustments_clone = current_edit_adjustments.clone().unwrap();
        hydrate_adjustments(&state, &mut adjustments_clone);

        let new_transform_hash = calculate_transform_hash(&adjustments_clone);
        let cached_preview_lock = state.cached_preview.lock().unwrap();
        let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);

        let (preview_image, scale, unscaled_crop_offset) = if let Some(cached) =
            &*cached_preview_lock
        {
            if cached.transform_hash == new_transform_hash && cached.preview_dim == preview_dim {
                let img = Arc::clone(&cached.image);
                let s = cached.scale;
                let offset = cached.unscaled_crop_offset;
                drop(cached_preview_lock);
                let owned_img = Arc::try_unwrap(img).unwrap_or_else(|arc| (*arc).clone());
                (owned_img, s, offset)
            } else {
                drop(cached_preview_lock);
                generate_transformed_preview(
                    &state,
                    &loaded_image,
                    &adjustments_clone,
                    preview_dim,
                )?
            }
        } else {
            drop(cached_preview_lock);
            generate_transformed_preview(&state, &loaded_image, &adjustments_clone, preview_dim)?
        };

        let (img_w, img_h) = preview_image.dimensions();
        let mask_definitions: Vec<MaskDefinition> = adjustments_clone
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let scaled_crop_offset = (
            unscaled_crop_offset.0 * scale,
            unscaled_crop_offset.1 * scale,
        );

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    img_w,
                    img_h,
                    scale,
                    scaled_crop_offset,
                    &adjustments_clone,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut all_adjustments =
            get_all_adjustments_from_json(&adjustments_clone, is_raw, tm_override);
        all_adjustments.global.show_clipping = 0;

        let lut = adjustments_clone["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&loaded_image.path, &adjustments_clone).wrapping_add(1);

        let request = RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
        };
        let processed_preview =
            if output_format.eq_ignore_ascii_case("jxl") && export_settings.jxl_bit_depth == 16 {
                process_and_get_high_precision_dynamic_image(
                    &context,
                    &state,
                    &preview_image,
                    unique_hash,
                    request,
                    "estimate_export_size",
                )?
            } else {
                process_and_get_dynamic_image(
                    &context,
                    &state,
                    &preview_image,
                    unique_hash,
                    request,
                    "estimate_export_size",
                )?
            };
        let processed_preview = if output_format.eq_ignore_ascii_case("jxl") {
            normalize_jxl_alpha_semantics(
                processed_preview,
                !is_raw && loaded_image.image.color().has_alpha(),
            )
        } else {
            processed_preview
        };

        let preview_bytes = encode_image_to_bytes(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.jxl_bit_depth,
            export_settings.jxl_effort,
            None,
        )?;
        let preview_byte_size = preview_bytes.len();

        let (transformed_full_res, _) =
            apply_all_transformations(&loaded_image.image, &adjustments_clone);
        let (full_w, full_h) = transformed_full_res.dimensions();

        let (final_full_w, final_full_h) = if let Some(resize_opts) = &export_settings.resize {
            calculate_resize_target(full_w, full_h, resize_opts)
        } else {
            (full_w, full_h)
        };

        let (processed_preview_w, processed_preview_h) = processed_preview.dimensions();
        let pixel_ratio = if processed_preview_w > 0 && processed_preview_h > 0 {
            (final_full_w as f64 * final_full_h as f64)
                / (processed_preview_w as f64 * processed_preview_h as f64)
        } else {
            1.0
        };

        (preview_byte_size as f64 * pixel_ratio) as usize
    } else {
        let metadata = crate::exif_processing::load_sidecar(&sidecar_path);
        let mut js_adjustments = metadata.adjustments;

        const ESTIMATE_DIM: u32 = 1280;

        let file_slice: Vec<u8>;
        let mmap_guard;
        let file_data: &[u8] = match read_file_mapped(Path::new(&source_path_str)) {
            Ok(mmap) => {
                mmap_guard = Some(mmap);
                mmap_guard.as_ref().unwrap()
            }
            Err(_) => {
                file_slice = fs::read(&source_path_str).map_err(|io_err| io_err.to_string())?;
                &file_slice
            }
        };

        let original_image =
            load_base_image_from_bytes(file_data, &source_path_str, true, &settings, None)
                .map_err(|e| e.to_string())?;

        let raw_scale_factor = if is_raw {
            crate::raw_processing::get_fast_demosaic_scale_factor(
                file_data,
                original_image.width(),
                original_image.height(),
            )
        } else {
            1.0
        };

        if let Some(crop_val) = js_adjustments.get_mut("crop")
            && let Ok(c) = serde_json::from_value::<Crop>(crop_val.clone())
        {
            *crop_val = serde_json::to_value(Crop {
                x: c.x * raw_scale_factor as f64,
                y: c.y * raw_scale_factor as f64,
                width: c.width * raw_scale_factor as f64,
                height: c.height * raw_scale_factor as f64,
            })
            .unwrap_or(serde_json::Value::Null);
        }

        let (transformed_shrunk_res, unscaled_crop_offset) =
            apply_all_transformations(Cow::Borrowed(&original_image), &js_adjustments);
        let (shrunk_w, shrunk_h) = transformed_shrunk_res.dimensions();

        let preview_base = if shrunk_w > ESTIMATE_DIM || shrunk_h > ESTIMATE_DIM {
            downscale_f32_image(transformed_shrunk_res.as_ref(), ESTIMATE_DIM, ESTIMATE_DIM)
        } else {
            transformed_shrunk_res.into_owned()
        };

        let (preview_w, preview_h) = preview_base.dimensions();
        let gpu_scale = if shrunk_w > 0 {
            preview_w as f32 / shrunk_w as f32
        } else {
            1.0
        };
        let total_scale = gpu_scale * raw_scale_factor;

        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();
        let scaled_crop_offset = (
            unscaled_crop_offset.0 * gpu_scale,
            unscaled_crop_offset.1 * gpu_scale,
        );

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    preview_w,
                    preview_h,
                    total_scale,
                    scaled_crop_offset,
                    &js_adjustments,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut all_adjustments =
            get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);
        all_adjustments.global.show_clipping = 0;

        let lut = js_adjustments["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&source_path_str, &js_adjustments).wrapping_add(1);

        let request = RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
        };
        let processed_preview =
            if output_format.eq_ignore_ascii_case("jxl") && export_settings.jxl_bit_depth == 16 {
                process_and_get_high_precision_dynamic_image(
                    &context,
                    &state,
                    &preview_base,
                    unique_hash,
                    request,
                    "estimate_batch_export_size",
                )?
            } else {
                process_and_get_dynamic_image(
                    &context,
                    &state,
                    &preview_base,
                    unique_hash,
                    request,
                    "estimate_batch_export_size",
                )?
            };
        let processed_preview = if output_format.eq_ignore_ascii_case("jxl") {
            normalize_jxl_alpha_semantics(
                processed_preview,
                !is_raw && original_image.color().has_alpha(),
            )
        } else {
            processed_preview
        };

        let preview_bytes = encode_image_to_bytes(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.jxl_bit_depth,
            export_settings.jxl_effort,
            None,
        )?;
        let single_image_estimated_size = preview_bytes.len();

        let full_w = (shrunk_w as f32 / raw_scale_factor).round() as u32;
        let full_h = (shrunk_h as f32 / raw_scale_factor).round() as u32;

        let (final_full_w, final_full_h) = if let Some(resize_opts) = &export_settings.resize {
            calculate_resize_target(full_w, full_h, resize_opts)
        } else {
            (full_w, full_h)
        };

        let (processed_preview_w, processed_preview_h) = processed_preview.dimensions();
        let pixel_ratio = if processed_preview_w > 0 && processed_preview_h > 0 {
            (final_full_w as f64 * final_full_h as f64)
                / (processed_preview_w as f64 * processed_preview_h as f64)
        } else {
            1.0
        };

        (single_image_estimated_size as f64 * pixel_ratio) as usize
    };

    Ok(single_image_extrapolated_size * paths.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, Rgba, RgbaImage};
    use jxl_oxide::JxlImage;

    fn decode_jxl(bytes: &[u8]) -> JxlImage {
        JxlImage::builder()
            .read(Cursor::new(bytes))
            .expect("synthetic JXL should decode")
    }

    fn decoded_u16(image: &JxlImage, max: f32) -> Vec<u16> {
        image
            .render_frame(0)
            .expect("frame should render")
            .image_all_channels()
            .buf()
            .iter()
            .map(|sample| (sample.clamp(0.0, 1.0) * max).round() as u16)
            .collect()
    }

    fn assert_jxl_contract(image: &JxlImage, width: u32, height: u32, depth: u32, channels: usize) {
        assert_eq!(image.width(), width);
        assert_eq!(image.height(), height);
        assert_eq!(
            image.image_header().metadata.bit_depth.bits_per_sample(),
            depth
        );
        assert_eq!(image.image_header().metadata.orientation, 1);
        assert_eq!(image.pixel_format().channels(), channels);
    }

    #[test]
    fn old_export_settings_default_jxl_fields() {
        let settings: ExportSettings = serde_json::from_value(serde_json::json!({
            "jpegQuality": 90,
            "resize": null,
            "keepMetadata": false,
            "stripGps": true,
            "filenameTemplate": null,
            "watermark": null
        }))
        .expect("old settings should remain readable");

        assert_eq!(settings.jxl_bit_depth, 8);
        assert_eq!(settings.jxl_effort, 5);
    }

    #[test]
    fn jxl16_memory_budget_scales_with_large_pixel_counts() {
        let twelve_gib_free = 12 * GIB;
        let (threads, task_bytes, safe_budget) =
            memory_limited_export_threads(8, 8, twelve_gib_free, 100_000_000, true)
                .expect("100 MP should be admitted as one task with 12 GiB free");
        assert_eq!(threads, 1);
        assert!(task_bytes > 7 * GIB);
        assert_eq!(safe_budget, twelve_gib_free * 85 / 100);

        let error = memory_limited_export_threads(8, 8, twelve_gib_free, 200_000_000, true)
            .expect_err("200 MP must fail before allocation when it exceeds the safe budget");
        assert!(error.contains("200.0 MP"));
    }

    #[test]
    fn jxl16_memory_budget_allows_bounded_parallelism_when_headroom_exists() {
        let (threads, _, _) =
            memory_limited_export_threads(8, 8, 24 * GIB, 100_000_000, true).unwrap();
        assert_eq!(threads, 2);
    }

    #[test]
    fn standard_export_keeps_the_existing_minimum_task_budget() {
        assert_eq!(
            estimated_export_task_bytes(1_000_000, false),
            MIN_EXPORT_TASK_BYTES
        );
    }

    #[test]
    fn sixteen_bit_quantization_clamps_rounds_and_rejects_non_finite() {
        assert_eq!(quantize_unit_f32_to_u16(-0.25).unwrap(), 0);
        assert_eq!(quantize_unit_f32_to_u16(0.0).unwrap(), 0);
        assert_eq!(quantize_unit_f32_to_u16(0.5).unwrap(), 32768);
        assert_eq!(quantize_unit_f32_to_u16(1.0).unwrap(), u16::MAX);
        assert_eq!(quantize_unit_f32_to_u16(1.25).unwrap(), u16::MAX);
        assert!(quantize_unit_f32_to_u16(f32::NAN).is_err());
        assert!(quantize_unit_f32_to_u16(f32::INFINITY).is_err());
        assert!(quantize_unit_f32_to_u16(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn lossless_rgb8_jxl_roundtrip_is_pixel_exact() {
        let pixels = vec![0, 1, 2, 17, 128, 254, 255, 33, 99, 8, 7, 6];
        let image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(2, 2, pixels.clone()).expect("valid RGB8 fixture"),
        );

        for effort in [4, 5, 7] {
            let encoded = encode_image_to_bytes(&image, "jxl", 100, 8, effort, None).unwrap();
            let decoded = decode_jxl(&encoded);
            assert_jxl_contract(&decoded, 2, 2, 8, 3);
            let actual: Vec<u8> = decoded_u16(&decoded, u8::MAX as f32)
                .into_iter()
                .map(|sample| sample as u8)
                .collect();
            assert_eq!(actual, pixels, "effort {effort} changed pixels");
            verify_jxl_export(&encoded, 2, 2, 8, 3, false, true, None).unwrap();
        }
    }

    #[test]
    fn lossless_jxl_uses_selected_effort_on_the_shared_pool() {
        for effort in [4, 5, 7] {
            let config = lossless_jxl_config(effort);
            assert_eq!(config.effort(), effort);
            assert_eq!(config.threads(), 0);
        }
    }

    #[test]
    fn jxl_exif_is_embedded_during_encoding_and_roundtrips() {
        let temp = tempfile::tempdir().unwrap();
        let missing_source = temp.path().join("missing-source.jpg");
        let rendered = exif_processing::RenderedImageMetadata {
            width: 2,
            height: 2,
            bits_per_sample: 16,
            samples_per_pixel: 3,
        };
        let exif_tiff = exif_processing::build_jxl_exif_tiff(
            missing_source.to_str().unwrap(),
            None,
            true,
            rendered,
        )
        .unwrap();
        let image =
            DynamicImage::ImageRgb32F(ImageBuffer::from_pixel(2, 2, Rgb([0.125, 0.5, 0.875])));

        let encoded = encode_image_to_bytes(&image, "jxl", 100, 16, 5, Some(&exif_tiff)).unwrap();
        verify_jxl_export(&encoded, 2, 2, 16, 3, true, true, Some(&exif_tiff)).unwrap();
    }

    #[test]
    fn jxl_exif_verification_reads_the_complete_trailing_box() {
        let temp = tempfile::tempdir().unwrap();
        let missing_source = temp.path().join("missing-source.jpg");
        let rendered = exif_processing::RenderedImageMetadata {
            width: 128,
            height: 128,
            bits_per_sample: 8,
            samples_per_pixel: 3,
        };
        let expected_exif = exif_processing::build_jxl_exif_tiff(
            missing_source.to_str().unwrap(),
            None,
            true,
            rendered,
        )
        .unwrap();
        let mut padded_exif = expected_exif.clone();
        padded_exif.extend_from_slice(&vec![0; 8 * 1024]);
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(128, 128, |x, y| {
            Rgb([
                (x.wrapping_mul(17) ^ y.wrapping_mul(31)) as u8,
                (x.wrapping_mul(11).wrapping_add(y.wrapping_mul(23))) as u8,
                (x.wrapping_mul(29) ^ y.wrapping_mul(7)) as u8,
            ])
        }));

        let encoded = encode_image_to_bytes(&image, "jxl", 99, 8, 4, Some(&padded_exif)).unwrap();
        let extracted = extract_jxl_exif_tiff(&encoded)
            .unwrap()
            .expect("the complete Exif box should be found");
        assert_eq!(extracted, padded_exif);

        // jxl-oxide 0.12 stops reading once the image codestream completes, so
        // a large trailing Exif box is finalized from only the bytes already in
        // its 4 KiB input buffer. This documents the decoder behavior that
        // caused real exports to fail with `Truncated field value`.
        let decoder_view = decode_jxl(&encoded);
        let decoder_exif_len = match decoder_view.aux_boxes().first_exif().unwrap() {
            jxl_oxide::AuxBoxData::Data(raw) => raw.payload().len(),
            other => panic!("expected decoder Exif data, got {other:?}"),
        };
        assert!(decoder_exif_len < padded_exif.len());

        verify_jxl_export(&encoded, 128, 128, 8, 3, true, true, Some(&expected_exif)).unwrap();
    }

    #[test]
    fn export_cancellation_check_is_idempotent() {
        let cancellation_token = AtomicBool::new(false);
        ensure_export_not_cancelled(&cancellation_token).unwrap();
        cancellation_token.store(true, Ordering::SeqCst);
        assert_eq!(
            ensure_export_not_cancelled(&cancellation_token).unwrap_err(),
            "Export cancelled"
        );
        assert_eq!(
            ensure_export_not_cancelled(&cancellation_token).unwrap_err(),
            "Export cancelled"
        );
    }

    #[test]
    fn jxl_exif_verification_accepts_byte_different_semantic_match() {
        let temp = tempfile::tempdir().unwrap();
        let missing_source = temp.path().join("missing-source.jpg");
        let rendered = exif_processing::RenderedImageMetadata {
            width: 2,
            height: 2,
            bits_per_sample: 16,
            samples_per_pixel: 3,
        };
        let expected_exif = exif_processing::build_jxl_exif_tiff(
            missing_source.to_str().unwrap(),
            None,
            true,
            rendered,
        )
        .unwrap();
        let mut padded_exif = expected_exif.clone();
        padded_exif.extend_from_slice(&[0; 16]);
        let image =
            DynamicImage::ImageRgb32F(ImageBuffer::from_pixel(2, 2, Rgb([0.125, 0.5, 0.875])));

        let encoded = encode_image_to_bytes(&image, "jxl", 100, 16, 5, Some(&padded_exif)).unwrap();
        verify_jxl_export(&encoded, 2, 2, 16, 3, true, true, Some(&expected_exif)).unwrap();
    }

    #[test]
    fn jxl_exif_verification_rejects_extra_semantic_tag() {
        let temp = tempfile::tempdir().unwrap();
        let missing_source = temp.path().join("missing-source.jpg");
        let rendered = exif_processing::RenderedImageMetadata {
            width: 2,
            height: 2,
            bits_per_sample: 8,
            samples_per_pixel: 3,
        };
        let expected_exif = exif_processing::build_jxl_exif_tiff(
            missing_source.to_str().unwrap(),
            None,
            true,
            rendered,
        )
        .unwrap();

        let mut sidecar = crate::image_processing::ImageMetadata::default();
        sidecar.exif = Some(HashMap::from([(
            "Make".to_string(),
            "Unexpected Camera".to_string(),
        )]));
        fs::write(
            exif_processing::get_primary_sidecar_path(&missing_source),
            serde_json::to_vec(&sidecar).unwrap(),
        )
        .unwrap();
        let actual_exif = exif_processing::build_jxl_exif_tiff(
            missing_source.to_str().unwrap(),
            None,
            true,
            rendered,
        )
        .unwrap();
        let image = DynamicImage::new_rgb8(2, 2);
        let encoded = encode_image_to_bytes(&image, "jxl", 100, 8, 5, Some(&actual_exif)).unwrap();

        let error =
            verify_jxl_export(&encoded, 2, 2, 8, 3, true, true, Some(&expected_exif)).unwrap_err();
        assert!(error.contains("semantic verification failed"));
    }

    #[test]
    fn lossless_rgb16_jxl_preserves_sub_eight_bit_precision() {
        let expected = vec![
            1000u16, 1001, 1002, 32767, 32768, 32769, 65000, 65001, 65002, 0, 1, 65535,
        ];
        assert_eq!(expected[0] / 257, expected[1] / 257);
        let pixels: Vec<f32> = expected
            .iter()
            .map(|sample| *sample as f32 / u16::MAX as f32)
            .collect();
        let image = DynamicImage::ImageRgb32F(
            ImageBuffer::<Rgb<f32>, _>::from_raw(2, 2, pixels)
                .expect("valid high precision fixture"),
        );

        let encoded = encode_image_to_bytes(&image, "jxl", 100, 16, 5, None).unwrap();
        let decoded = decode_jxl(&encoded);
        assert_jxl_contract(&decoded, 2, 2, 16, 3);
        assert_eq!(decoded_u16(&decoded, u16::MAX as f32), expected);
    }

    #[test]
    fn opaque_rgba_semantics_are_preserved_at_both_depths() {
        let image = DynamicImage::ImageRgba32F(
            ImageBuffer::<Rgba<f32>, _>::from_raw(
                2,
                1,
                vec![0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0],
            )
            .unwrap(),
        );

        for depth in [8, 16] {
            let encoded = encode_image_to_bytes(&image, "jxl", 100, depth, 5, None).unwrap();
            let decoded = decode_jxl(&encoded);
            assert_jxl_contract(&decoded, 2, 1, u32::from(depth), 4);
        }
    }

    #[test]
    fn lossy_eight_and_sixteen_bit_jxl_decode_with_declared_contract() {
        let values: Vec<f32> = (0..16 * 16 * 3)
            .map(|index| ((index * 37) % 65536) as f32 / u16::MAX as f32)
            .collect();
        let image = DynamicImage::ImageRgb32F(
            ImageBuffer::<Rgb<f32>, _>::from_raw(16, 16, values).unwrap(),
        );

        for depth in [8, 16] {
            let encoded = encode_image_to_bytes(&image, "jxl", 92, depth, 5, None).unwrap();
            let decoded = decode_jxl(&encoded);
            assert_jxl_contract(&decoded, 16, 16, u32::from(depth), 3);
            decoded
                .render_frame(0)
                .expect("lossy frame should fully decode");
        }
    }

    #[test]
    fn jxl_rejects_unsupported_effort_in_both_codec_modes() {
        let image = DynamicImage::new_rgb8(2, 2);
        for quality in [100, 92] {
            let error = encode_image_to_bytes(&image, "jxl", quality, 8, 6, None).unwrap_err();
            assert!(error.contains("expected 4, 5, or 7"));
        }
    }

    #[test]
    fn sixteen_bit_export_rejects_non_finite_render_output() {
        let image = DynamicImage::ImageRgb32F(
            ImageBuffer::<Rgb<f32>, _>::from_raw(1, 1, vec![0.0, f32::NAN, 1.0]).unwrap(),
        );

        let error = encode_image_to_bytes(&image, "jxl", 100, 16, 5, None).unwrap_err();
        assert!(error.contains("non-finite"));
    }

    #[test]
    fn watermark_blending_keeps_high_precision_base_pixels() {
        let temp = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .expect("create watermark fixture");
        RgbaImage::from_pixel(1, 1, Rgba([200, 100, 50, 128]))
            .save(temp.path())
            .expect("write watermark fixture");

        let mut image = DynamicImage::ImageRgb32F(ImageBuffer::from_pixel(
            4,
            4,
            Rgb([0.12345, 0.23456, 0.34567]),
        ));
        apply_watermark(
            &mut image,
            &WatermarkSettings {
                path: temp.path().to_string_lossy().into_owned(),
                anchor: WatermarkAnchor::TopLeft,
                scale: 25.0,
                spacing: 0.0,
                opacity: 100.0,
            },
        )
        .unwrap();

        let DynamicImage::ImageRgb32F(buffer) = image else {
            panic!("watermarking must retain the high precision image variant");
        };
        assert_eq!(buffer.get_pixel(1, 1).0, [0.12345, 0.23456, 0.34567]);
        assert_ne!(
            buffer.get_pixel(0, 0)[0],
            (buffer.get_pixel(0, 0)[0] * 255.0).round() / 255.0,
            "blended pixel should not be pre-quantized to eight bits"
        );
    }
}
