use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Cursor};
use std::path::{Path, PathBuf};

use crate::formats::is_raw_file;
use crate::image_processing::ImageMetadata;
use chrono::{DateTime, NaiveDateTime, Utc};
use exif::{Exif, In, Value};
use little_exif::exif_tag::ExifTag;
use little_exif::filetype::FileExtension;
use little_exif::ifd::ExifTagGroup;
use little_exif::metadata::Metadata;
use little_exif::rational::{iR64, uR64};
use rawler::decoders::RawMetadata;

pub fn truncate_large_exif(value: &str) -> String {
    if value.len() <= 500 {
        return value.to_string();
    }

    let mut start_idx = 200;
    while !value.is_char_boundary(start_idx) {
        start_idx -= 1;
    }

    let mut end_idx = value.len() - 200;
    while !value.is_char_boundary(end_idx) {
        end_idx += 1;
    }

    if start_idx < end_idx {
        let start_str = &value[..start_idx];
        let end_str = &value[end_idx..];
        return format!("{}...{}", start_str, end_str);
    }

    value.to_string()
}

pub fn load_sidecar(sidecar_path: &Path) -> ImageMetadata {
    if !sidecar_path.exists() {
        return ImageMetadata::default();
    }

    let Ok(content) = fs::read_to_string(sidecar_path) else {
        return ImageMetadata::default();
    };

    let mut meta = serde_json::from_str::<ImageMetadata>(&content).unwrap_or_default();
    let mut healed = false;

    if let Some(ref mut exif_map) = meta.exif {
        for val in exif_map.values_mut() {
            if val.len() > 500 {
                *val = truncate_large_exif(val);
                healed = true;
            }
        }
    }

    if healed && let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = fs::write(sidecar_path, json);
        log::info!(
            "Auto-healed bloated sidecar for: {}",
            sidecar_path.display()
        );
    }

    meta
}

fn to_ur64(val: &exif::Rational) -> uR64 {
    uR64 {
        nominator: val.num,
        denominator: val.denom,
    }
}

fn to_ir64(val: &exif::SRational) -> iR64 {
    iR64 {
        nominator: val.num,
        denominator: val.denom,
    }
}

fn clean_creation_datetime_str(s: &str) -> &str {
    s.trim().trim_matches('"').trim_matches('\'').trim()
}

fn fmt_date_str(s: String) -> String {
    if let Some(dt) = parse_creation_datetime(&s) {
        return dt.format("%Y-%m-%d %H:%M:%S").to_string();
    }
    clean_creation_datetime_str(&s).to_string()
}

fn normalize_creation_datetime(s: &str) -> Option<String> {
    let normalized = s.replace('T', " ");
    let (date, time) = normalized.split_once(' ')?;
    Some(format!("{} {}", date.replace(':', "-"), time))
}

fn parse_creation_datetime(s: &str) -> Option<NaiveDateTime> {
    let clean = clean_creation_datetime_str(s);
    if clean.is_empty() {
        return None;
    }

    let normalized = normalize_creation_datetime(clean);
    for candidate in std::iter::once(clean).chain(normalized.as_deref()) {
        for format in [
            "%Y:%m:%d %H:%M:%S",
            "%Y:%m:%d %H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
        ] {
            if let Ok(dt) = NaiveDateTime::parse_from_str(candidate, format) {
                return Some(dt);
            }
        }
    }

    None
}

fn parse_creation_field(field: &exif::Field) -> Option<DateTime<Utc>> {
    parse_creation_datetime(&field.display_value().to_string())
        .map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
}

fn parse_raw_creation_date(date_str: Option<&str>) -> Option<DateTime<Utc>> {
    parse_creation_datetime(date_str?).map(|dt| DateTime::from_naive_utc_and_offset(dt, Utc))
}

pub fn read_exif(file_bytes: &[u8]) -> Option<Exif> {
    let exifreader = exif::Reader::new();
    exifreader
        .read_from_container(&mut Cursor::new(file_bytes))
        .ok()
}

pub fn read_raw_metadata(file_bytes: &[u8]) -> Option<RawMetadata> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let loader = rawler::RawLoader::new();
        let raw_source = rawler::rawsource::RawSource::new_from_slice(file_bytes);
        let decoder = loader.get_decoder(&raw_source).ok()?;
        decoder.raw_metadata(&raw_source, &Default::default()).ok()
    })) {
        Ok(metadata) => metadata,
        Err(_) => {
            log::warn!("RAW metadata decoder panicked");
            None
        }
    }
}

pub fn read_exposure_time_secs(path: &str, file_bytes: &[u8]) -> Option<f32> {
    if let Some(map) = read_rrexif_sidecar(Path::new(path))
        && let Some(val_str) = map.get("ExposureTime").or(map.get("ShutterSpeedValue"))
    {
        let cleaned = val_str.replace(" s", "");
        if cleaned.contains('/') {
            let parts: Vec<&str> = cleaned.split('/').collect();
            if parts.len() == 2
                && let (Ok(num), Ok(den)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>())
                && den != 0.0
            {
                return Some(num / den);
            }
        } else if let Ok(val) = cleaned.parse::<f32>() {
            return Some(val);
        }
    }

    if is_raw_file(path)
        && let Some(meta) = read_raw_metadata(file_bytes)
    {
        if let Some(r) = meta.exif.exposure_time {
            return if r.d == 0 {
                None
            } else {
                Some(r.n as f32 / r.d as f32)
            };
        } else if let Some(r) = meta.exif.shutter_speed_value {
            return if r.d == 0 {
                None
            } else {
                Some(r.n as f32 / r.d as f32)
            };
        }
    }

    if let Some(exif) = read_exif(file_bytes) {
        if let Some(exposure) = exif.get_field(exif::Tag::ExposureTime, In::PRIMARY) {
            if let Value::Rational(ref r) = exposure.value {
                if r.is_empty() {
                    return None;
                }

                let val = r.first()?;

                return if val.denom == 0 {
                    None
                } else {
                    Some(val.num as f32 / val.denom as f32)
                };
            }
        } else if let Some(shutter_speed) =
            exif.get_field(exif::Tag::ShutterSpeedValue, In::PRIMARY)
            && let Value::Rational(ref r) = shutter_speed.value
        {
            if r.is_empty() {
                return None;
            }

            let val = r.first()?;

            return if val.denom == 0 {
                None
            } else {
                Some(val.num as f32 / val.denom as f32)
            };
        }
    }
    None
}

pub fn read_iso(path: &str, file_bytes: &[u8]) -> Option<u32> {
    if let Some(map) = read_rrexif_sidecar(Path::new(path))
        && let Some(val_str) = map
            .get("ISOSpeed")
            .or(map.get("PhotographicSensitivity"))
            .or(map.get("ISOSpeedRatings"))
        && let Ok(val) = val_str.parse::<u32>()
    {
        return Some(val);
    }

    if is_raw_file(path)
        && let Some(meta) = read_raw_metadata(file_bytes)
    {
        if let Some(r) = meta.exif.iso_speed {
            return Some(r);
        } else if let Some(r) = meta.exif.iso_speed_ratings {
            return Some(r as u32);
        }
    }

    if let Some(exif) = read_exif(file_bytes) {
        if let Some(r) = exif.get_field(exif::Tag::ISOSpeed, In::PRIMARY) {
            return r.value.get_uint(0);
        } else if let Some(r) = exif.get_field(exif::Tag::PhotographicSensitivity, In::PRIMARY) {
            return r.value.get_uint(0);
        }
    }
    None
}

pub fn extract_metadata(file_bytes: &[u8]) -> Option<HashMap<String, String>> {
    let mut map = HashMap::new();

    if let Some(exif_obj) = read_exif(file_bytes) {
        for field in exif_obj.fields() {
            match field.tag {
                exif::Tag::ExposureTime => {
                    if let exif::Value::Rational(ref v) = field.value
                        && !v.is_empty()
                    {
                        let r = &v[0];
                        if r.num == 1 && r.denom > 1 {
                            map.insert("ExposureTime".to_string(), format!("1/{} s", r.denom));
                        } else {
                            let val = r.num as f32 / r.denom as f32;
                            if val < 1.0 && val > 0.0 {
                                map.insert(
                                    "ExposureTime".to_string(),
                                    format!("1/{} s", (1.0 / val).round()),
                                );
                            } else {
                                map.insert("ExposureTime".to_string(), format!("{} s", val));
                            }
                        }
                    }
                }
                exif::Tag::ShutterSpeedValue => {
                    if let exif::Value::SRational(ref v) = field.value
                        && !v.is_empty()
                    {
                        let val = v[0].num as f32 / v[0].denom as f32;
                        map.insert("ShutterSpeedValue".to_string(), val.to_string());
                    }
                }
                exif::Tag::FNumber => {
                    if let exif::Value::Rational(ref v) = field.value
                        && !v.is_empty()
                    {
                        let val = v[0].num as f32 / v[0].denom as f32;
                        map.insert("FNumber".to_string(), format!("f/{}", val));
                    }
                }
                exif::Tag::ApertureValue => {
                    if let exif::Value::Rational(ref v) = field.value
                        && !v.is_empty()
                    {
                        let val = v[0].num as f32 / v[0].denom as f32;
                        map.insert("ApertureValue".to_string(), format!("f/{}", val));
                    }
                }
                exif::Tag::FocalLength => {
                    if let exif::Value::Rational(ref v) = field.value
                        && !v.is_empty()
                    {
                        let val = v[0].num as f32 / v[0].denom as f32;
                        map.insert("FocalLength".to_string(), val.to_string());
                        map.insert("FocalLengthIn35mmFilm".to_string(), val.to_string());
                    }
                }
                exif::Tag::PhotographicSensitivity | exif::Tag::ISOSpeed => {
                    map.insert(
                        "PhotographicSensitivity".to_string(),
                        field.display_value().to_string(),
                    );
                    map.insert("ISOSpeed".to_string(), field.display_value().to_string());
                }
                exif::Tag::DateTimeOriginal => {
                    map.insert(
                        "DateTimeOriginal".to_string(),
                        fmt_date_str(field.display_value().to_string()),
                    );
                }
                exif::Tag::DateTime => {
                    map.insert(
                        "CreateDate".to_string(),
                        fmt_date_str(field.display_value().to_string()),
                    );
                }
                exif::Tag::DateTimeDigitized => {
                    map.insert(
                        "ModifyDate".to_string(),
                        fmt_date_str(field.display_value().to_string()),
                    );
                }
                _ => {
                    let val = field.display_value().with_unit(&exif_obj).to_string();
                    if !val.trim().is_empty() {
                        map.insert(field.tag.to_string(), val);
                    }
                }
            }
        }
    }

    if !map.is_empty() {
        return Some(map);
    }

    let metadata = read_raw_metadata(file_bytes)?;

    let exif = metadata.exif;

    let fmt_rat = |r: &rawler::formats::tiff::Rational| -> f32 {
        if r.d == 0 {
            0.0
        } else {
            r.n as f32 / r.d as f32
        }
    };

    let fmt_srat = |r: &rawler::formats::tiff::SRational| -> f32 {
        if r.d == 0 {
            0.0
        } else {
            r.n as f32 / r.d as f32
        }
    };

    let mut insert_if_present = |key: &str, val: String| {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            map.insert(key.to_string(), truncate_large_exif(trimmed));
        }
    };

    insert_if_present("Make", metadata.make);
    insert_if_present("Model", metadata.model);

    if let Some(v) = exif.artist {
        insert_if_present("Artist", v);
    }
    if let Some(v) = exif.copyright {
        insert_if_present("Copyright", v);
    }
    if let Some(v) = exif.owner_name {
        insert_if_present("OwnerName", v);
    }
    if let Some(v) = exif.serial_number {
        insert_if_present("SerialNumber", v);
    }
    if let Some(v) = exif.image_number {
        insert_if_present("ImageNumber", v.to_string());
    }
    if let Some(v) = exif.user_comment {
        insert_if_present("UserComment", v);
    }

    if let Some(v) = exif.date_time_original {
        insert_if_present("DateTimeOriginal", fmt_date_str(v));
    }
    if let Some(v) = exif.create_date {
        insert_if_present("CreateDate", fmt_date_str(v));
    }
    if let Some(v) = exif.modify_date {
        insert_if_present("ModifyDate", fmt_date_str(v));
    }

    if let Some(v) = exif.offset_time {
        insert_if_present("OffsetTime", v);
    }
    if let Some(v) = exif.offset_time_original {
        insert_if_present("OffsetTimeOriginal", v);
    }
    if let Some(v) = exif.offset_time_digitized {
        insert_if_present("OffsetTimeDigitized", v);
    }
    if let Some(v) = exif.sub_sec_time {
        insert_if_present("SubSecTime", v);
    }
    if let Some(v) = exif.sub_sec_time_original {
        insert_if_present("SubSecTimeOriginal", v);
    }
    if let Some(v) = exif.sub_sec_time_digitized {
        insert_if_present("SubSecTimeDigitized", v);
    }

    if let Some(v) = exif.lens_model {
        insert_if_present("LensModel", v);
    } else if let Some(lens_desc) = &metadata.lens {
        insert_if_present("LensModel", lens_desc.lens_model.clone());
    }

    if let Some(v) = exif.lens_make {
        insert_if_present("LensMake", v);
    } else if let Some(lens_desc) = &metadata.lens {
        insert_if_present("LensMake", lens_desc.lens_make.clone());
    }

    if let Some(v) = exif.lens_serial_number {
        insert_if_present("LensSerialNumber", v);
    }

    if let Some(v) = exif.orientation {
        insert_if_present("Orientation", v.to_string());
    }

    if let Some(r) = exif.fnumber {
        let val = fmt_rat(&r);
        insert_if_present("FNumber", format!("f/{}", val));
    }

    if let Some(r) = exif.aperture_value {
        let val = fmt_rat(&r);
        insert_if_present("ApertureValue", format!("f/{}", val));
    }

    if let Some(r) = exif.max_aperture_value {
        insert_if_present("MaxApertureValue", fmt_rat(&r).to_string());
    }

    if let Some(r) = exif.exposure_time {
        if r.n == 1 && r.d > 1 {
            insert_if_present("ExposureTime", format!("1/{} s", r.d));
        } else {
            let val = fmt_rat(&r);
            if val < 1.0 && val > 0.0 {
                insert_if_present("ExposureTime", format!("1/{} s", (1.0 / val).round()));
            } else {
                insert_if_present("ExposureTime", format!("{} s", val));
            }
        }
    }

    if let Some(r) = exif.shutter_speed_value {
        insert_if_present("ShutterSpeedValue", fmt_srat(&r).to_string());
    }

    if let Some(v) = exif.iso_speed {
        insert_if_present("PhotographicSensitivity", v.to_string());
        insert_if_present("ISOSpeed", v.to_string());
    } else if let Some(v) = exif.iso_speed_ratings {
        insert_if_present("PhotographicSensitivity", v.to_string());
        insert_if_present("ISOSpeedRatings", v.to_string());
    }

    if let Some(v) = exif.recommended_exposure_index {
        insert_if_present("RecommendedExposureIndex", v.to_string());
    }
    if let Some(v) = exif.sensitivity_type {
        insert_if_present("SensitivityType", v.to_string());
    }

    if let Some(r) = exif.focal_length {
        let val = fmt_rat(&r);
        insert_if_present("FocalLength", val.to_string());
        insert_if_present("FocalLengthIn35mmFilm", val.to_string());
    }

    if let Some(r) = exif.exposure_bias {
        insert_if_present("ExposureBiasValue", fmt_srat(&r).to_string());
    }

    if let Some(v) = exif.metering_mode {
        insert_if_present("MeteringMode", v.to_string());
    }
    if let Some(v) = exif.light_source {
        insert_if_present("LightSource", v.to_string());
    }
    if let Some(v) = exif.flash {
        insert_if_present("Flash", v.to_string());
    }
    if let Some(v) = exif.white_balance {
        insert_if_present("WhiteBalance", v.to_string());
    }
    if let Some(v) = exif.exposure_program {
        insert_if_present("ExposureProgram", v.to_string());
    }
    if let Some(v) = exif.exposure_mode {
        insert_if_present("ExposureMode", v.to_string());
    }
    if let Some(v) = exif.scene_capture_type {
        insert_if_present("SceneCaptureType", v.to_string());
    }
    if let Some(v) = exif.color_space {
        insert_if_present("ColorSpace", v.to_string());
    }
    if let Some(r) = exif.flash_energy {
        insert_if_present("FlashEnergy", fmt_rat(&r).to_string());
    }
    if let Some(r) = exif.brightness_value {
        insert_if_present("BrightnessValue", fmt_srat(&r).to_string());
    }

    if let Some(r) = exif.subject_distance {
        insert_if_present("SubjectDistance", fmt_rat(&r).to_string());
    }
    if let Some(v) = exif.subject_distance_range {
        insert_if_present("SubjectDistanceRange", v.to_string());
    }

    if let Some(gps) = exif.gps {
        let fmt_gps_coord = |coords: &[rawler::formats::tiff::Rational; 3]| -> String {
            format!(
                "{} deg {} min {} sec",
                fmt_rat(&coords[0]),
                fmt_rat(&coords[1]),
                fmt_rat(&coords[2])
            )
        };

        if let Some(lat) = gps.gps_latitude {
            insert_if_present("GPSLatitude", fmt_gps_coord(&lat));
        }
        if let Some(lat_ref) = gps.gps_latitude_ref {
            insert_if_present("GPSLatitudeRef", lat_ref);
        }
        if let Some(lon) = gps.gps_longitude {
            insert_if_present("GPSLongitude", fmt_gps_coord(&lon));
        }
        if let Some(lon_ref) = gps.gps_longitude_ref {
            insert_if_present("GPSLongitudeRef", lon_ref);
        }
        if let Some(alt) = gps.gps_altitude {
            insert_if_present("GPSAltitude", fmt_rat(&alt).to_string());
        }
        if let Some(alt_ref) = gps.gps_altitude_ref {
            insert_if_present("GPSAltitudeRef", alt_ref.to_string());
        }
        if let Some(v) = gps.gps_img_direction {
            insert_if_present("GPSImgDirection", fmt_rat(&v).to_string());
        }
        if let Some(v) = gps.gps_img_direction_ref {
            insert_if_present("GPSImgDirectionRef", v);
        }
        if let Some(v) = gps.gps_speed {
            insert_if_present("GPSSpeed", fmt_rat(&v).to_string());
        }
        if let Some(v) = gps.gps_speed_ref {
            insert_if_present("GPSSpeedRef", v);
        }
        if let Some(v) = gps.gps_status {
            insert_if_present("GPSStatus", v);
        }
        if let Some(v) = gps.gps_measure_mode {
            insert_if_present("GPSMeasureMode", v);
        }
        if let Some(v) = gps.gps_dop {
            insert_if_present("GPSDOP", fmt_rat(&v).to_string());
        }
        if let Some(v) = gps.gps_map_datum {
            insert_if_present("GPSMapDatum", v);
        }
    }

    Some(map)
}

pub fn get_creation_date_from_path(path: &Path) -> DateTime<Utc> {
    if let Some(map) = read_rrexif_sidecar(path)
        && let Some(dt_str) = map.get("DateTimeOriginal").or(map.get("CreateDate"))
        && let Some(dt) = parse_creation_datetime(dt_str)
    {
        return DateTime::from_naive_utc_and_offset(dt, Utc);
    }

    if let Ok(file) = std::fs::File::open(path) {
        let mut bufreader = BufReader::new(&file);
        let exifreader = exif::Reader::new();

        if let Ok(exif_obj) = exifreader.read_from_container(&mut bufreader) {
            for tag in [exif::Tag::DateTimeOriginal, exif::Tag::DateTime] {
                if let Some(field) = exif_obj.get_field(tag, exif::In::PRIMARY)
                    && let Some(dt) = parse_creation_field(field)
                {
                    return dt;
                }
            }
        }
    }

    if is_raw_file(path.to_string_lossy().as_ref()) {
        let loader = rawler::RawLoader::new();
        if let Ok(raw_source) = rawler::rawsource::RawSource::new(path)
            && let Ok(decoder) = loader.get_decoder(&raw_source)
            && let Ok(metadata) = decoder.raw_metadata(&raw_source, &Default::default())
        {
            if let Some(dt) = parse_raw_creation_date(metadata.exif.date_time_original.as_deref()) {
                return dt;
            }
            if let Some(dt) = parse_raw_creation_date(metadata.exif.create_date.as_deref()) {
                return dt;
            }
        }
    }

    fs::metadata(path)
        .ok()
        .and_then(|m| m.created().ok())
        .map(DateTime::<Utc>::from)
        .unwrap_or_else(Utc::now)
}

#[cfg(target_os = "android")]
pub fn get_creation_date_from_bytes(path_hint: &str, file_bytes: &[u8]) -> DateTime<Utc> {
    if let Some(exif_obj) = read_exif(file_bytes) {
        for tag in [exif::Tag::DateTimeOriginal, exif::Tag::DateTime] {
            if let Some(field) = exif_obj.get_field(tag, exif::In::PRIMARY)
                && let Some(dt) = parse_creation_field(field)
            {
                return dt;
            }
        }
    }

    if is_raw_file(path_hint)
        && let Some(metadata) = read_raw_metadata(file_bytes)
    {
        if let Some(dt) = parse_raw_creation_date(metadata.exif.date_time_original.as_deref()) {
            return dt;
        }
        if let Some(dt) = parse_raw_creation_date(metadata.exif.create_date.as_deref()) {
            return dt;
        }
    }

    Utc::now()
}

fn parse_sidecar_ur64(value: &str) -> Option<uR64> {
    let cleaned = value
        .replace("f/", "")
        .replace(" s", "")
        .replace(" mm", "")
        .replace(['\"', '\''], "");
    let value = cleaned.trim();

    if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator.trim().parse::<u32>().ok()?;
        let denominator = denominator.trim().parse::<u32>().ok()?;
        return (denominator != 0).then_some(uR64 {
            nominator: numerator,
            denominator,
        });
    }

    let value = value.parse::<f64>().ok()?;
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 {
        return None;
    }
    const SCALE: u32 = 1_000_000;
    Some(uR64 {
        nominator: (value * SCALE as f64).round().clamp(0.0, u32::MAX as f64) as u32,
        denominator: SCALE,
    })
}

fn parse_sidecar_ir64(value: &str) -> Option<iR64> {
    let cleaned = value.replace(['\"', '\''], "");
    let value = cleaned.trim();
    if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator.trim().parse::<i32>().ok()?;
        let denominator = denominator.trim().parse::<i32>().ok()?;
        return (denominator != 0).then_some(iR64 {
            nominator: numerator,
            denominator,
        });
    }

    let value = value.parse::<f64>().ok()?;
    if !value.is_finite() || value < i32::MIN as f64 || value > i32::MAX as f64 {
        return None;
    }
    const SCALE: i32 = 1_000_000;
    Some(iR64 {
        nominator: (value * SCALE as f64)
            .round()
            .clamp(i32::MIN as f64, i32::MAX as f64) as i32,
        denominator: SCALE,
    })
}

fn parse_sidecar_gps_coordinate(value: &str) -> Option<Vec<uR64>> {
    let values: Vec<f64> = value
        .split(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+')))
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<f64>().ok())
        .collect();

    let (degrees, minutes, seconds) = match values.as_slice() {
        [decimal] => {
            let decimal = decimal.abs();
            let degrees = decimal.floor();
            let minute_fraction = (decimal - degrees) * 60.0;
            (
                degrees,
                minute_fraction.floor(),
                minute_fraction.fract() * 60.0,
            )
        }
        [degrees, minutes, seconds, ..] => (degrees.abs(), minutes.abs(), seconds.abs()),
        _ => return None,
    };

    [degrees, minutes, seconds]
        .into_iter()
        .map(|component| parse_sidecar_ur64(&component.to_string()))
        .collect()
}

fn remove_sidecar_tag(metadata: &mut Metadata, key: &str) {
    if matches!(
        key,
        "ISOSpeed" | "PhotographicSensitivity" | "ISOSpeedRatings"
    ) {
        metadata.remove_tag(ExifTag::ISO(Vec::new()));
        metadata.remove_tag(ExifTag::ISOSpeed(Vec::new()));
        return;
    }
    let tag = match key {
        "DateTimeOriginal" => ExifTag::DateTimeOriginal(String::new()),
        "SubSecTimeOriginal" => ExifTag::SubSecTimeOriginal(String::new()),
        "OffsetTimeOriginal" => ExifTag::OffsetTimeOriginal(String::new()),
        "Make" => ExifTag::Make(String::new()),
        "Model" => ExifTag::Model(String::new()),
        "LensMake" => ExifTag::LensMake(String::new()),
        "LensModel" => ExifTag::LensModel(String::new()),
        "FNumber" => ExifTag::FNumber(Vec::new()),
        "FocalLength" => ExifTag::FocalLength(Vec::new()),
        "ExposureTime" => ExifTag::ExposureTime(Vec::new()),
        "ExposureBiasValue" => ExifTag::ExposureCompensation(Vec::new()),
        "Artist" => ExifTag::Artist(String::new()),
        "Copyright" => ExifTag::Copyright(String::new()),
        "GPSLatitude" => ExifTag::GPSLatitude(Vec::new()),
        "GPSLatitudeRef" => ExifTag::GPSLatitudeRef(String::new()),
        "GPSLongitude" => ExifTag::GPSLongitude(Vec::new()),
        "GPSLongitudeRef" => ExifTag::GPSLongitudeRef(String::new()),
        "GPSAltitude" => ExifTag::GPSAltitude(Vec::new()),
        "GPSAltitudeRef" => ExifTag::GPSAltitudeRef(Vec::new()),
        _ => return,
    };
    metadata.remove_tag(tag);
}

fn apply_sidecar_field(
    metadata: &mut Metadata,
    key: &str,
    value: &str,
    strip_gps: bool,
    only_if_missing: bool,
    strict: bool,
) -> Result<(), String> {
    let clean_string = || value.replace(['\"', '\''], "").trim().to_string();
    let is_missing = |tag: &ExifTag| metadata.get_tag(tag).next().is_none();
    let should_set = |tag: &ExifTag| !only_if_missing || is_missing(tag);
    let parse_error = || {
        if strict {
            Err(format!("Invalid sidecar EXIF value for {key}"))
        } else {
            Ok(())
        }
    };

    match key {
        "DateTimeOriginal" => {
            let tag = ExifTag::DateTimeOriginal(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::DateTimeOriginal(clean_string()));
            }
        }
        "SubSecTimeOriginal" => {
            let tag = ExifTag::SubSecTimeOriginal(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::SubSecTimeOriginal(clean_string()));
            }
        }
        "OffsetTimeOriginal" => {
            let tag = ExifTag::OffsetTimeOriginal(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::OffsetTimeOriginal(clean_string()));
            }
        }
        "Make" => {
            let tag = ExifTag::Make(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::Make(clean_string()));
            }
        }
        "Model" => {
            let tag = ExifTag::Model(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::Model(clean_string()));
            }
        }
        "LensMake" => {
            let tag = ExifTag::LensMake(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::LensMake(clean_string()));
            }
        }
        "LensModel" => {
            let tag = ExifTag::LensModel(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::LensModel(clean_string()));
            }
        }
        "Artist" => {
            let tag = ExifTag::Artist(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::Artist(clean_string()));
            }
        }
        "Copyright" => {
            let tag = ExifTag::Copyright(String::new());
            if should_set(&tag) {
                metadata.set_tag(ExifTag::Copyright(clean_string()));
            }
        }
        "FNumber" | "FocalLength" | "ExposureTime" => {
            let empty_tag = match key {
                "FNumber" => ExifTag::FNumber(Vec::new()),
                "FocalLength" => ExifTag::FocalLength(Vec::new()),
                _ => ExifTag::ExposureTime(Vec::new()),
            };
            if should_set(&empty_tag) {
                let Some(parsed) = parse_sidecar_ur64(value) else {
                    return parse_error();
                };
                metadata.set_tag(match key {
                    "FNumber" => ExifTag::FNumber(vec![parsed]),
                    "FocalLength" => ExifTag::FocalLength(vec![parsed]),
                    _ => ExifTag::ExposureTime(vec![parsed]),
                });
            }
        }
        "ISOSpeed" | "PhotographicSensitivity" | "ISOSpeedRatings" => {
            let short_tag = ExifTag::ISO(Vec::new());
            let long_tag = ExifTag::ISOSpeed(Vec::new());
            if !only_if_missing || (is_missing(&short_tag) && is_missing(&long_tag)) {
                let Ok(iso) = clean_string().parse::<u32>() else {
                    return parse_error();
                };
                if let Ok(short_iso) = u16::try_from(iso) {
                    metadata.set_tag(ExifTag::ISO(vec![short_iso]));
                } else {
                    metadata.set_tag(ExifTag::ISOSpeed(vec![iso]));
                }
            }
        }
        "ExposureBiasValue" => {
            let tag = ExifTag::ExposureCompensation(Vec::new());
            if should_set(&tag) {
                let Some(bias) = parse_sidecar_ir64(value) else {
                    return parse_error();
                };
                metadata.set_tag(ExifTag::ExposureCompensation(vec![bias]));
            }
        }
        "GPSLatitude" | "GPSLongitude" if !strip_gps => {
            let empty_tag = if key == "GPSLatitude" {
                ExifTag::GPSLatitude(Vec::new())
            } else {
                ExifTag::GPSLongitude(Vec::new())
            };
            if should_set(&empty_tag) {
                let Some(coordinate) = parse_sidecar_gps_coordinate(value) else {
                    return parse_error();
                };
                metadata.set_tag(if key == "GPSLatitude" {
                    ExifTag::GPSLatitude(coordinate)
                } else {
                    ExifTag::GPSLongitude(coordinate)
                });
            }
        }
        "GPSLatitudeRef" | "GPSLongitudeRef" if !strip_gps => {
            let empty_tag = if key == "GPSLatitudeRef" {
                ExifTag::GPSLatitudeRef(String::new())
            } else {
                ExifTag::GPSLongitudeRef(String::new())
            };
            if should_set(&empty_tag) {
                metadata.set_tag(if key == "GPSLatitudeRef" {
                    ExifTag::GPSLatitudeRef(clean_string())
                } else {
                    ExifTag::GPSLongitudeRef(clean_string())
                });
            }
        }
        "GPSAltitude" if !strip_gps => {
            let tag = ExifTag::GPSAltitude(Vec::new());
            if should_set(&tag) {
                let Some(altitude) = parse_sidecar_ur64(value) else {
                    return parse_error();
                };
                metadata.set_tag(ExifTag::GPSAltitude(vec![altitude]));
            }
        }
        "GPSAltitudeRef" if !strip_gps => {
            let tag = ExifTag::GPSAltitudeRef(Vec::new());
            if should_set(&tag) {
                let Ok(reference) = clean_string().parse::<u8>() else {
                    return parse_error();
                };
                metadata.set_tag(ExifTag::GPSAltitudeRef(vec![reference]));
            }
        }
        _ => {}
    }
    Ok(())
}

fn apply_sidecar_map(
    metadata: &mut Metadata,
    map: &HashMap<String, String>,
    strip_gps: bool,
    only_if_missing: bool,
    strict: bool,
) -> Result<(), String> {
    for (key, value) in map {
        apply_sidecar_field(metadata, key, value, strip_gps, only_if_missing, strict)?;
    }
    Ok(())
}

fn apply_sidecar_overrides(
    metadata: &mut Metadata,
    overrides: &HashMap<String, Option<String>>,
    strip_gps: bool,
) -> Result<(), String> {
    for (key, value) in overrides {
        match value {
            Some(value) => apply_sidecar_field(metadata, key, value, strip_gps, false, true)?,
            None => remove_sidecar_tag(metadata, key),
        }
    }
    Ok(())
}

fn read_legacy_rrexif_sidecar_read_only(
    image_path: &Path,
) -> Result<Option<HashMap<String, String>>, String> {
    let legacy = get_rrexif_path(image_path);
    if !legacy.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&legacy).map_err(|error| {
        format!(
            "Failed to read EXIF sidecar '{}': {error}",
            legacy.display()
        )
    })?;
    serde_json::from_str(&content).map(Some).map_err(|error| {
        format!(
            "Failed to parse EXIF sidecar '{}': {error}",
            legacy.display()
        )
    })
}

fn load_jxl_sidecar_read_only(path: &Path) -> Result<Option<ImageMetadata>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path).map_err(|error| {
        format!(
            "Failed to read JXL metadata sidecar '{}': {error}",
            path.display()
        )
    })?;
    serde_json::from_str(&content).map(Some).map_err(|error| {
        format!(
            "Failed to parse JXL metadata sidecar '{}': {error}",
            path.display()
        )
    })
}

fn read_jxl_container_exif_for_export(path: &Path, strict: bool) -> Result<Option<Exif>, String> {
    let image = match jxl_oxide::JxlImage::open_with_defaults(path) {
        Ok(image) => image,
        Err(error) if strict => {
            return Err(format!(
                "Failed to parse source JXL '{}': {error}",
                path.display()
            ));
        }
        Err(_) => return Ok(None),
    };

    let raw_exif = match image.aux_boxes().first_exif() {
        Ok(raw_exif) => raw_exif,
        Err(error) if strict => {
            return Err(format!(
                "Failed to parse source JXL Exif box '{}': {error}",
                path.display()
            ));
        }
        Err(_) => return Ok(None),
    };

    match raw_exif {
        jxl_oxide::AuxBoxData::NotFound => Ok(None),
        jxl_oxide::AuxBoxData::Decoding if strict => Err(format!(
            "Source JXL Exif box is incomplete ('{}')",
            path.display()
        )),
        jxl_oxide::AuxBoxData::Decoding => Ok(None),
        jxl_oxide::AuxBoxData::Data(raw_exif) => {
            let offset = raw_exif.tiff_header_offset() as usize;
            let Some(tiff) = raw_exif.payload().get(offset..) else {
                return if strict {
                    Err(format!(
                        "Source JXL Exif TIFF offset is invalid ('{}')",
                        path.display()
                    ))
                } else {
                    Ok(None)
                };
            };
            match exif::Reader::new().read_raw(tiff.to_vec()) {
                Ok(exif) => Ok(Some(exif)),
                Err(error) if strict => Err(format!(
                    "Failed to parse source JXL EXIF '{}': {error}",
                    path.display()
                )),
                Err(_) => Ok(None),
            }
        }
    }
}

fn read_container_exif_for_export(path: &Path, strict: bool) -> Result<Option<Exif>, String> {
    if !path.exists() {
        return Ok(None);
    }

    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension == "jxl" {
        return read_jxl_container_exif_for_export(path, strict);
    }

    // kamadak-exif only understands these container families. RapidRAW also
    // accepts formats such as BMP, GIF, EXR and QOI; those valid inputs simply
    // have no typed container EXIF for this reader and must not be treated as
    // corrupt when JXL metadata retention is enabled.
    if !matches!(
        extension.as_str(),
        "jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp" | "heif" | "heic" | "avif"
    ) {
        return Ok(None);
    }

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if strict => {
            return Err(format!(
                "Failed to open source metadata '{}': {error}",
                path.display()
            ));
        }
        Err(_) => return Ok(None),
    };
    let mut reader = std::io::BufReader::new(file);
    match exif::Reader::new().read_from_container(&mut reader) {
        Ok(exif) => Ok(Some(exif)),
        Err(exif::Error::NotFound(_)) => Ok(None),
        Err(error) if strict => Err(format!(
            "Failed to parse source EXIF '{}': {error}",
            path.display()
        )),
        Err(_) => Ok(None),
    }
}

fn read_raw_metadata_for_export(path: &Path, strict: bool) -> Result<Option<RawMetadata>, String> {
    if !path.exists() {
        return Ok(None);
    }

    if crate::image_loader::cached_raw_metadata_failure_for_path(path) {
        return if strict {
            Err(format!(
                "Skipping cached RAW metadata failure ('{}')",
                path.display()
            ))
        } else {
            Ok(None)
        };
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<RawMetadata, String> {
            let loader = rawler::RawLoader::new();
            let raw_source = rawler::rawsource::RawSource::new(path)
                .map_err(|error| format!("Failed to open RAW metadata: {error}"))?;
            let decoder = loader
                .get_decoder(&raw_source)
                .map_err(|error| format!("Failed to create RAW metadata decoder: {error}"))?;
            decoder
                .raw_metadata(&raw_source, &Default::default())
                .map_err(|error| format!("Failed to parse RAW metadata: {error}"))
        },
    ));

    let result = match result {
        Ok(result) => result,
        Err(_) => {
            crate::image_loader::remember_raw_metadata_panic_for_path(path);
            Err("RAW metadata decoder panicked".to_string())
        }
    };

    match result {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if strict => Err(format!("{} ('{}')", error, path.display())),
        Err(_) => Ok(None),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RenderedImageMetadata {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
}

pub fn write_image_with_metadata(
    image_bytes: &mut Vec<u8>,
    original_path_str: &str,
    output_format: &str,
    keep_metadata: bool,
    strip_gps: bool,
    rendered: Option<RenderedImageMetadata>,
    source_sidecar_path: Option<&Path>,
) -> Result<(), String> {
    let output_format = output_format.to_lowercase();
    let is_jxl = output_format == "jxl";

    // FIXME: temporary solution until I find a way to write metadata to TIFF
    if !keep_metadata || output_format == "tiff" {
        return Ok(());
    }

    let original_path = Path::new(original_path_str);
    if !original_path.exists() && !is_jxl {
        return Ok(());
    }

    // Skip TIFF sources to avoid potential tag corruption issues
    let original_ext = original_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if (original_ext == "tiff" || original_ext == "tif") && !is_jxl {
        return Ok(());
    }

    let file_type = match output_format.as_str() {
        "jpg" | "jpeg" => FileExtension::JPEG,
        "jxl" => FileExtension::JXL,
        "png" => FileExtension::PNG {
            as_zTXt_chunk: true,
        },
        "tiff" => FileExtension::TIFF,
        _ => return Ok(()),
    };

    let mut metadata = Metadata::new();
    let mut source_read_success = false;

    if !is_jxl && let Some(map) = read_rrexif_sidecar(original_path) {
        source_read_success = true;

        let clean_s = |s: &String| s.replace('"', "").trim().to_string();

        let parse_ur64 = |s: &str| -> Option<uR64> {
            let cleaned_string = s
                .replace("f/", "")
                .replace(" s", "")
                .replace(" mm", "")
                .replace("\"", "");

            let val = cleaned_string.trim();

            if val.contains('/') {
                let parts: Vec<&str> = val.split('/').collect();
                if parts.len() == 2
                    && let (Ok(n), Ok(d)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                {
                    return Some(uR64 {
                        nominator: n,
                        denominator: d,
                    });
                }
            } else if let Ok(f) = val.parse::<f32>() {
                return Some(uR64 {
                    nominator: (f * 1000.0) as u32,
                    denominator: 1000,
                });
            }
            None
        };
        if let Some(val) = map.get("Make") {
            metadata.set_tag(ExifTag::Make(clean_s(val)));
        }
        if let Some(val) = map.get("Model") {
            metadata.set_tag(ExifTag::Model(clean_s(val)));
        }
        if let Some(val) = map.get("LensMake") {
            metadata.set_tag(ExifTag::LensMake(clean_s(val)));
        }
        if let Some(val) = map.get("LensModel") {
            metadata.set_tag(ExifTag::LensModel(clean_s(val)));
        }
        if let Some(val) = map.get("Artist") {
            metadata.set_tag(ExifTag::Artist(clean_s(val)));
        }
        if let Some(val) = map.get("Copyright") {
            metadata.set_tag(ExifTag::Copyright(clean_s(val)));
        }
        if let Some(val) = map.get("UserComment") {
            metadata.set_tag(ExifTag::UserComment(clean_s(val).into_bytes()));
        }
        if let Some(val) = map.get("ImageDescription") {
            metadata.set_tag(ExifTag::ImageDescription(clean_s(val)));
        }
        if let Some(val) = map.get("DateTimeOriginal") {
            metadata.set_tag(ExifTag::DateTimeOriginal(clean_s(val)));
        }
        if let Some(val) = map.get("CreateDate") {
            metadata.set_tag(ExifTag::CreateDate(clean_s(val)));
        }
        if let Some(val) = map.get("FNumber")
            && let Some(ur) = parse_ur64(val)
        {
            metadata.set_tag(ExifTag::FNumber(vec![ur]));
        }
        if let Some(val) = map.get("ExposureTime")
            && let Some(ur) = parse_ur64(val)
        {
            metadata.set_tag(ExifTag::ExposureTime(vec![ur]));
        }
        if let Some(val) = map.get("FocalLength")
            && let Some(ur) = parse_ur64(val)
        {
            metadata.set_tag(ExifTag::FocalLength(vec![ur]));
        }
        if let Some(val) = map.get("FocalLengthIn35mmFilm") {
            let cleaned = val.replace(" mm", "").replace("\"", "");
            let trimmed = cleaned.trim();
            if let Ok(f_val) = trimmed.parse::<f32>() {
                metadata.set_tag(ExifTag::FocalLengthIn35mmFormat(vec![f_val.round() as u16]));
            }
        }
        if let Some(val) = map.get("ISOSpeed").or(map.get("PhotographicSensitivity"))
            && let Ok(iso) = val.replace('"', "").trim().parse::<u16>()
        {
            metadata.set_tag(ExifTag::ISO(vec![iso]));
        }
    }

    if !source_read_success
        && (!is_jxl || !is_raw_file(original_path_str))
        && let Some(exif_obj) = read_container_exif_for_export(original_path, is_jxl)?
    {
        source_read_success = true;

        let get_string_val = |field: &exif::Field| -> String {
            match &field.value {
                exif::Value::Ascii(vec) => vec
                    .iter()
                    .map(|v| {
                        String::from_utf8_lossy(v)
                            .trim_matches(char::from(0))
                            .to_string()
                    })
                    .collect::<Vec<String>>()
                    .join(" "),
                _ => field
                    .display_value()
                    .to_string()
                    .replace("\"", "")
                    .trim()
                    .to_string(),
            }
        };

        if let Some(f) = exif_obj.get_field(exif::Tag::Make, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::Make(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::Model, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::Model(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::LensMake, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::LensMake(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::LensModel, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::LensModel(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::Artist, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::Artist(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::Copyright, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::Copyright(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::DateTimeOriginal(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::SubSecTimeOriginal, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::SubSecTimeOriginal(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::OffsetTimeOriginal, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::OffsetTimeOriginal(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::DateTime, exif::In::PRIMARY) {
            metadata.set_tag(ExifTag::CreateDate(get_string_val(f)));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::FNumber, exif::In::PRIMARY)
            && let exif::Value::Rational(v) = &f.value
            && !v.is_empty()
        {
            metadata.set_tag(ExifTag::FNumber(vec![to_ur64(&v[0])]));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::ExposureTime, exif::In::PRIMARY)
            && let exif::Value::Rational(v) = &f.value
            && !v.is_empty()
        {
            metadata.set_tag(ExifTag::ExposureTime(vec![to_ur64(&v[0])]));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::FocalLength, exif::In::PRIMARY)
            && let exif::Value::Rational(v) = &f.value
            && !v.is_empty()
        {
            metadata.set_tag(ExifTag::FocalLength(vec![to_ur64(&v[0])]));
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::ExposureBiasValue, exif::In::PRIMARY) {
            match &f.value {
                exif::Value::SRational(v) if !v.is_empty() => {
                    metadata.set_tag(ExifTag::ExposureCompensation(vec![to_ir64(&v[0])]));
                }
                exif::Value::Rational(v) if !v.is_empty() => {
                    metadata.set_tag(ExifTag::ExposureCompensation(vec![iR64 {
                        nominator: v[0].num as i32,
                        denominator: v[0].denom as i32,
                    }]));
                }
                _ => {}
            }
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::PhotographicSensitivity, exif::In::PRIMARY) {
            if let Some(val) = f.value.get_uint(0) {
                if let Ok(short_iso) = u16::try_from(val) {
                    metadata.set_tag(ExifTag::ISO(vec![short_iso]));
                } else {
                    metadata.set_tag(ExifTag::ISOSpeed(vec![val]));
                }
            }
        } else if let Some(f) = exif_obj.get_field(exif::Tag::ISOSpeed, exif::In::PRIMARY)
            && let Some(val) = f.value.get_uint(0)
        {
            if let Ok(short_iso) = u16::try_from(val) {
                metadata.set_tag(ExifTag::ISO(vec![short_iso]));
            } else {
                metadata.set_tag(ExifTag::ISOSpeed(vec![val]));
            }
        }
        if let Some(f) = exif_obj.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)
            && let Some(val) = f.value.get_uint(0)
        {
            metadata.set_tag(ExifTag::FocalLengthIn35mmFormat(vec![val as u16]));
        }
        if !strip_gps {
            if let Some(f) = exif_obj.get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                && let exif::Value::Rational(v) = &f.value
                && v.len() >= 3
            {
                metadata.set_tag(ExifTag::GPSLatitude(vec![
                    to_ur64(&v[0]),
                    to_ur64(&v[1]),
                    to_ur64(&v[2]),
                ]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::GPSLatitudeRef, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::GPSLatitudeRef(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::GPSLongitude, exif::In::PRIMARY)
                && let exif::Value::Rational(v) = &f.value
                && v.len() >= 3
            {
                metadata.set_tag(ExifTag::GPSLongitude(vec![
                    to_ur64(&v[0]),
                    to_ur64(&v[1]),
                    to_ur64(&v[2]),
                ]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::GPSLongitudeRef, exif::In::PRIMARY) {
                metadata.set_tag(ExifTag::GPSLongitudeRef(get_string_val(f)));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::GPSAltitude, exif::In::PRIMARY)
                && let exif::Value::Rational(v) = &f.value
                && !v.is_empty()
            {
                metadata.set_tag(ExifTag::GPSAltitude(vec![to_ur64(&v[0])]));
            }
            if let Some(f) = exif_obj.get_field(exif::Tag::GPSAltitudeRef, exif::In::PRIMARY) {
                let alt_ref = f.value.get_uint(0).unwrap_or(0) as u8;
                metadata.set_tag(ExifTag::GPSAltitudeRef(vec![alt_ref]));
            }
        }
    }

    if !source_read_success
        && is_raw_file(original_path_str)
        && let Some(meta) = read_raw_metadata_for_export(original_path, is_jxl)?
    {
        if !meta.make.is_empty() {
            metadata.set_tag(ExifTag::Make(meta.make.clone()));
        }
        if !meta.model.is_empty() {
            metadata.set_tag(ExifTag::Model(meta.model.clone()));
        }
        let exif = meta.exif;
        if let Some(artist) = exif.artist {
            metadata.set_tag(ExifTag::Artist(artist));
        }
        if let Some(copyright) = exif.copyright {
            metadata.set_tag(ExifTag::Copyright(copyright));
        }
        if let Some(dt) = exif.date_time_original {
            metadata.set_tag(ExifTag::DateTimeOriginal(dt));
        }
        if let Some(subsec) = exif.sub_sec_time_original {
            metadata.set_tag(ExifTag::SubSecTimeOriginal(subsec));
        }
        if let Some(offset) = exif.offset_time_original {
            metadata.set_tag(ExifTag::OffsetTimeOriginal(offset));
        }
        if let Some(dt) = exif.create_date {
            metadata.set_tag(ExifTag::CreateDate(dt));
        }
        if let Some(lens_make) = exif.lens_make {
            metadata.set_tag(ExifTag::LensMake(lens_make));
        }
        if let Some(lens_model) = exif.lens_model {
            metadata.set_tag(ExifTag::LensModel(lens_model));
        }
        if let Some(f) = exif.fnumber {
            metadata.set_tag(ExifTag::FNumber(vec![uR64 {
                nominator: f.n,
                denominator: f.d,
            }]));
        }
        if let Some(t) = exif.exposure_time {
            metadata.set_tag(ExifTag::ExposureTime(vec![uR64 {
                nominator: t.n,
                denominator: t.d,
            }]));
        }
        if let Some(fl) = exif.focal_length {
            metadata.set_tag(ExifTag::FocalLength(vec![uR64 {
                nominator: fl.n,
                denominator: fl.d,
            }]));
        }
        if let Some(iso) = exif.iso_speed {
            if let Ok(short_iso) = u16::try_from(iso) {
                metadata.set_tag(ExifTag::ISO(vec![short_iso]));
            } else {
                metadata.set_tag(ExifTag::ISOSpeed(vec![iso]));
            }
        } else if let Some(iso) = exif.iso_speed_ratings {
            metadata.set_tag(ExifTag::ISO(vec![iso]));
        }
        if let Some(ev) = exif.exposure_bias {
            metadata.set_tag(ExifTag::ExposureCompensation(vec![iR64 {
                nominator: ev.n,
                denominator: ev.d,
            }]));
        }
        if let Some(flash) = exif.flash {
            metadata.set_tag(ExifTag::Flash(vec![flash]));
        }
        if let Some(metering) = exif.metering_mode {
            metadata.set_tag(ExifTag::MeteringMode(vec![metering]));
        }
        if let Some(wb) = exif.white_balance {
            metadata.set_tag(ExifTag::WhiteBalance(vec![wb]));
        }
        if let Some(prog) = exif.exposure_program {
            metadata.set_tag(ExifTag::ExposureProgram(vec![prog]));
        }
        if !strip_gps && let Some(gps) = exif.gps {
            if let Some(lat) = gps.gps_latitude {
                metadata.set_tag(ExifTag::GPSLatitude(vec![
                    uR64 {
                        nominator: lat[0].n,
                        denominator: lat[0].d,
                    },
                    uR64 {
                        nominator: lat[1].n,
                        denominator: lat[1].d,
                    },
                    uR64 {
                        nominator: lat[2].n,
                        denominator: lat[2].d,
                    },
                ]));
            }
            if let Some(lat_ref) = gps.gps_latitude_ref {
                metadata.set_tag(ExifTag::GPSLatitudeRef(lat_ref));
            }
            if let Some(lon) = gps.gps_longitude {
                metadata.set_tag(ExifTag::GPSLongitude(vec![
                    uR64 {
                        nominator: lon[0].n,
                        denominator: lon[0].d,
                    },
                    uR64 {
                        nominator: lon[1].n,
                        denominator: lon[1].d,
                    },
                    uR64 {
                        nominator: lon[2].n,
                        denominator: lon[2].d,
                    },
                ]));
            }
            if let Some(lon_ref) = gps.gps_longitude_ref {
                metadata.set_tag(ExifTag::GPSLongitudeRef(lon_ref));
            }
            if let Some(alt) = gps.gps_altitude {
                metadata.set_tag(ExifTag::GPSAltitude(vec![uR64 {
                    nominator: alt.n,
                    denominator: alt.d,
                }]));
            }
            if let Some(alt_ref) = gps.gps_altitude_ref {
                metadata.set_tag(ExifTag::GPSAltitudeRef(vec![alt_ref]));
            }
        }
    }

    if is_jxl {
        let primary_sidecar_path = get_primary_sidecar_path(original_path);
        let primary_sidecar = load_jxl_sidecar_read_only(&primary_sidecar_path)?;
        let mut primary_map_present = false;
        if let Some(primary_sidecar) = primary_sidecar.as_ref()
            && let Some(map) = primary_sidecar.exif.as_ref()
        {
            primary_map_present = true;
            apply_sidecar_map(&mut metadata, map, strip_gps, true, true)?;
        }
        if !primary_map_present
            && let Some(map) = read_legacy_rrexif_sidecar_read_only(original_path)?
        {
            apply_sidecar_map(&mut metadata, &map, strip_gps, true, true)?;
        }
        if let Some(overrides) = primary_sidecar
            .as_ref()
            .and_then(|sidecar| sidecar.exif_overrides.as_ref())
        {
            apply_sidecar_overrides(&mut metadata, overrides, strip_gps)?;
        }

        if let Some(source_sidecar_path) = source_sidecar_path
            && source_sidecar_path != primary_sidecar_path
            && let Some(virtual_sidecar) = load_jxl_sidecar_read_only(source_sidecar_path)?
        {
            if let Some(map) = virtual_sidecar.exif.as_ref() {
                apply_sidecar_map(&mut metadata, map, strip_gps, true, true)?;
            }
            if let Some(overrides) = virtual_sidecar.exif_overrides.as_ref() {
                apply_sidecar_overrides(&mut metadata, overrides, strip_gps)?;
            }
        }
    }

    metadata.set_tag(ExifTag::Software("RapidRAW".to_string()));
    metadata.set_tag(ExifTag::Orientation(vec![1u16]));
    metadata.set_tag(ExifTag::ColorSpace(vec![1u16]));

    if is_jxl {
        let rendered = rendered.ok_or_else(|| {
            "Rendered image metadata is required when writing JXL EXIF".to_string()
        })?;
        if !matches!(rendered.bits_per_sample, 8 | 16) {
            return Err(format!(
                "Unsupported JXL EXIF bit depth: {}",
                rendered.bits_per_sample
            ));
        }
        if !matches!(rendered.samples_per_pixel, 3 | 4) {
            return Err(format!(
                "Unsupported JXL EXIF channel count: {}",
                rendered.samples_per_pixel
            ));
        }

        metadata.set_tag(ExifTag::ImageWidth(vec![rendered.width]));
        metadata.set_tag(ExifTag::ImageHeight(vec![rendered.height]));
        metadata.set_tag(ExifTag::ExifImageWidth(vec![rendered.width]));
        metadata.set_tag(ExifTag::ExifImageHeight(vec![rendered.height]));
        metadata.set_tag(ExifTag::BitsPerSample(vec![
            rendered.bits_per_sample;
            rendered.samples_per_pixel
                as usize
        ]));
        metadata.set_tag(ExifTag::SamplesPerPixel(vec![rendered.samples_per_pixel]));
        metadata.set_tag(ExifTag::PhotometricInterpretation(vec![2u16]));
        let exported_at = chrono::Local::now();
        metadata.set_tag(ExifTag::ModifyDate(
            exported_at.format("%Y:%m:%d %H:%M:%S").to_string(),
        ));
        metadata.set_tag(ExifTag::OffsetTime(exported_at.format("%:z").to_string()));
    }

    if let Err(e) = metadata.write_to_vec(image_bytes, file_type) {
        if is_jxl {
            return Err(format!("Failed to write JXL EXIF metadata: {e}"));
        }
        log::warn!("Failed to write metadata: {}", e);
    }

    Ok(())
}

fn validate_rendered_jxl_exif(
    metadata: &Metadata,
    rendered: RenderedImageMetadata,
    strip_gps: bool,
) -> Result<(), String> {
    let has_tag = |needle: &ExifTag| metadata.get_tag(needle).next();

    match has_tag(&ExifTag::ImageWidth(Vec::new())) {
        Some(ExifTag::ImageWidth(value)) if value.as_slice() == [rendered.width] => {}
        _ => return Err("JXL EXIF ImageWidth verification failed".to_string()),
    }
    match has_tag(&ExifTag::ImageHeight(Vec::new())) {
        Some(ExifTag::ImageHeight(value)) if value.as_slice() == [rendered.height] => {}
        _ => return Err("JXL EXIF ImageHeight verification failed".to_string()),
    }
    match has_tag(&ExifTag::BitsPerSample(Vec::new())) {
        Some(ExifTag::BitsPerSample(value))
            if value.len() == rendered.samples_per_pixel as usize
                && value.iter().all(|value| *value == rendered.bits_per_sample) => {}
        _ => return Err("JXL EXIF BitsPerSample verification failed".to_string()),
    }
    match has_tag(&ExifTag::SamplesPerPixel(Vec::new())) {
        Some(ExifTag::SamplesPerPixel(value))
            if value.as_slice() == [rendered.samples_per_pixel] => {}
        _ => return Err("JXL EXIF SamplesPerPixel verification failed".to_string()),
    }
    match has_tag(&ExifTag::Orientation(Vec::new())) {
        Some(ExifTag::Orientation(value)) if value.as_slice() == [1] => {}
        _ => return Err("JXL EXIF Orientation verification failed".to_string()),
    }
    match has_tag(&ExifTag::ColorSpace(Vec::new())) {
        Some(ExifTag::ColorSpace(value)) if value.as_slice() == [1] => {}
        _ => return Err("JXL EXIF ColorSpace verification failed".to_string()),
    }
    match has_tag(&ExifTag::Software(String::new())) {
        Some(ExifTag::Software(value)) if value == "RapidRAW" => {}
        _ => return Err("JXL EXIF Software verification failed".to_string()),
    }

    if strip_gps
        && metadata
            .get_ifds()
            .iter()
            .any(|ifd| ifd.get_ifd_type() == ExifTagGroup::GPS)
    {
        return Err("JXL EXIF GPS stripping verification failed".to_string());
    }

    Ok(())
}

pub fn build_jxl_exif_tiff(
    original_path_str: &str,
    source_sidecar_path: Option<&Path>,
    strip_gps: bool,
    rendered: RenderedImageMetadata,
) -> Result<Vec<u8>, String> {
    // Build and round-trip a tiny metadata-only JXL container so the same
    // whitelist and source precedence rules are shared with JPEG metadata.
    // The returned pure TIFF payload is embedded by jxl-encoder while it
    // creates the final image container; the placeholder never contains pixels.
    // little_exif's container walker reads four bytes of box payload while
    // identifying auxiliary boxes, so keep this metadata-only codestream at
    // the minimum four bytes even though only the 0xff0a signature matters.
    let mut placeholder = vec![0xff, 0x0a, 0x00, 0x00];
    write_image_with_metadata(
        &mut placeholder,
        original_path_str,
        "jxl",
        true,
        strip_gps,
        Some(rendered),
        source_sidecar_path,
    )?;

    let metadata = Metadata::new_from_vec(&placeholder, FileExtension::JXL)
        .map_err(|error| format!("Failed to verify generated JXL EXIF: {error}"))?;
    validate_rendered_jxl_exif(&metadata, rendered, strip_gps)?;
    metadata
        .encode()
        .map_err(|error| format!("Failed to serialize JXL EXIF TIFF payload: {error}"))
}

pub fn get_primary_sidecar_path(image_path: &Path) -> PathBuf {
    let mut filename = image_path.file_name().unwrap_or_default().to_os_string();
    filename.push(".rrdata");
    image_path.with_file_name(filename)
}

pub fn get_rrexif_path(image_path: &Path) -> PathBuf {
    let mut filename = image_path.file_name().unwrap_or_default().to_os_string();
    filename.push(".rrexif");
    image_path.with_file_name(filename)
}

fn load_primary_metadata(image_path: &Path) -> ImageMetadata {
    let primary = get_primary_sidecar_path(image_path);
    load_sidecar(&primary)
}

fn save_primary_metadata(image_path: &Path, metadata: &ImageMetadata) -> std::io::Result<()> {
    let primary = get_primary_sidecar_path(image_path);
    let json = serde_json::to_string_pretty(metadata).map_err(std::io::Error::other)?;
    fs::write(&primary, json)
}

pub fn read_rrexif_sidecar(image_path: &Path) -> Option<HashMap<String, String>> {
    let metadata = load_primary_metadata(image_path);
    if let Some(exif) = metadata.exif {
        return Some(exif);
    }

    let legacy = get_rrexif_path(image_path);
    if legacy.exists()
        && let Ok(content) = fs::read_to_string(&legacy)
        && let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content)
    {
        let mut migrated = load_primary_metadata(image_path);
        migrated.exif = Some(map.clone());
        if save_primary_metadata(image_path, &migrated).is_ok() {
            let _ = fs::remove_file(&legacy);
        }
        return Some(map);
    }

    None
}

pub fn read_exif_data_from_bytes(path: &str, file_bytes: &[u8]) -> HashMap<String, String> {
    if is_raw_file(path)
        && let Some(map) = extract_metadata(file_bytes)
    {
        return map;
    }

    let mut exif_data = HashMap::new();
    if let Some(exif) = read_exif(file_bytes) {
        for field in exif.fields() {
            let raw_val = field.display_value().with_unit(&exif).to_string();
            exif_data.insert(field.tag.to_string(), truncate_large_exif(&raw_val));
        }
    }
    exif_data
}

pub fn read_exif_data(path: &str, file_bytes: &[u8]) -> HashMap<String, String> {
    let source_path = Path::new(path);
    if let Some(sidecar_exif) = read_rrexif_sidecar(source_path) {
        return sidecar_exif;
    }

    let exif_map = read_exif_data_from_bytes(path, file_bytes);
    if !exif_map.is_empty() {
        let mut metadata = load_primary_metadata(source_path);
        metadata.exif = Some(exif_map.clone());
        let _ = save_primary_metadata(source_path, &metadata);
    }
    exif_map
}

pub fn persist_exif_if_missing(source_path: &Path, source_path_str: &str, file_bytes: &[u8]) {
    {
        let metadata = load_primary_metadata(source_path);
        if metadata.exif.is_some() {
            return;
        }
    }

    let legacy = get_rrexif_path(source_path);
    if legacy.exists()
        && let Ok(content) = fs::read_to_string(&legacy)
        && let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&content)
    {
        let mut metadata = load_primary_metadata(source_path);
        metadata.exif = Some(map);
        if save_primary_metadata(source_path, &metadata).is_ok() {
            let _ = fs::remove_file(&legacy);
        }
        return;
    }

    let exif_map = read_exif_data_from_bytes(source_path_str, file_bytes);
    if exif_map.is_empty() {
        return;
    }

    let mut metadata = load_primary_metadata(source_path);

    if metadata.exif.is_none() {
        metadata.exif = Some(exif_map);
        let _ = save_primary_metadata(source_path, &metadata);
    }
}

pub fn write_rrexif_sidecar(source_path_str: &str, target_image_path: &Path) -> Result<(), String> {
    let source_path = Path::new(source_path_str);

    let exif_data = if let Some(existing) = read_rrexif_sidecar(source_path) {
        existing
    } else if let Ok(bytes) = fs::read(source_path) {
        read_exif_data_from_bytes(source_path_str, &bytes)
    } else {
        return Ok(());
    };

    if exif_data.is_empty() {
        return Ok(());
    }

    let mut metadata = load_primary_metadata(target_image_path);
    metadata.exif = Some(exif_data);
    save_primary_metadata(target_image_path, &metadata)
        .map_err(|e| format!("Failed to write sidecar: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use std::io::Cursor;

    fn rational(nominator: u32, denominator: u32) -> uR64 {
        uR64 {
            nominator,
            denominator,
        }
    }

    fn write_typed_jpeg_source(path: &Path) {
        let mut bytes = Vec::new();
        DynamicImage::new_rgb8(2, 2)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)
            .unwrap();

        let mut metadata = Metadata::new();
        metadata.set_tag(ExifTag::DateTimeOriginal("2026:01:02 03:04:05".to_string()));
        metadata.set_tag(ExifTag::SubSecTimeOriginal("123".to_string()));
        metadata.set_tag(ExifTag::OffsetTimeOriginal("+08:00".to_string()));
        metadata.set_tag(ExifTag::Make("Typed Camera Maker".to_string()));
        metadata.set_tag(ExifTag::Model("Typed Camera Model".to_string()));
        metadata.set_tag(ExifTag::LensMake("Typed Lens Maker".to_string()));
        metadata.set_tag(ExifTag::LensModel("Typed Lens Model".to_string()));
        metadata.set_tag(ExifTag::Artist("Typed Artist".to_string()));
        metadata.set_tag(ExifTag::Copyright("Typed Copyright".to_string()));
        metadata.set_tag(ExifTag::FNumber(vec![rational(28, 10)]));
        metadata.set_tag(ExifTag::FocalLength(vec![rational(50, 1)]));
        metadata.set_tag(ExifTag::ExposureTime(vec![rational(1, 125)]));
        metadata.set_tag(ExifTag::ISO(vec![400]));
        metadata.set_tag(ExifTag::ExposureCompensation(vec![iR64 {
            nominator: -1,
            denominator: 3,
        }]));
        metadata.set_tag(ExifTag::GPSLatitude(vec![
            rational(1, 1),
            rational(2, 1),
            rational(3, 1),
        ]));
        metadata.set_tag(ExifTag::GPSLatitudeRef("N".to_string()));
        metadata.set_tag(ExifTag::GPSLongitude(vec![
            rational(4, 1),
            rational(5, 1),
            rational(6, 1),
        ]));
        metadata.set_tag(ExifTag::GPSLongitudeRef("E".to_string()));
        metadata.set_tag(ExifTag::GPSAltitude(vec![rational(7, 1)]));
        metadata.set_tag(ExifTag::GPSAltitudeRef(vec![0]));
        metadata
            .write_to_vec(&mut bytes, FileExtension::JPEG)
            .unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn parse_tiff(bytes: Vec<u8>) -> Exif {
        exif::Reader::new().read_raw(bytes).unwrap()
    }

    fn ascii_value(exif: &Exif, tag: exif::Tag) -> Option<String> {
        let field = exif.get_field(tag, In::PRIMARY)?;
        match &field.value {
            Value::Ascii(values) => Some(
                values
                    .iter()
                    .map(|value| {
                        String::from_utf8_lossy(value)
                            .trim_matches(char::from(0))
                            .to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            _ => None,
        }
    }

    #[test]
    fn jxl_exif_uses_typed_source_then_explicit_overrides() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jpg");
        write_typed_jpeg_source(&source);

        let mut legacy = HashMap::new();
        legacy.insert("LensModel".to_string(), "Legacy Lens".to_string());
        legacy.insert("FNumber".to_string(), "f/9.9".to_string());
        let overrides = HashMap::from([
            ("LensModel".to_string(), Some("Explicit Lens".to_string())),
            ("Copyright".to_string(), None),
        ]);
        let mut sidecar = ImageMetadata::default();
        sidecar.exif = Some(legacy);
        sidecar.exif_overrides = Some(overrides);
        save_primary_metadata(&source, &sidecar).unwrap();

        let tiff = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            false,
            RenderedImageMetadata {
                width: 640,
                height: 480,
                bits_per_sample: 16,
                samples_per_pixel: 3,
            },
        )
        .unwrap();
        let exif = parse_tiff(tiff);

        assert_eq!(
            ascii_value(&exif, exif::Tag::LensModel).as_deref(),
            Some("Explicit Lens")
        );
        assert_eq!(
            ascii_value(&exif, exif::Tag::Make).as_deref(),
            Some("Typed Camera Maker")
        );
        assert_eq!(
            ascii_value(&exif, exif::Tag::Model).as_deref(),
            Some("Typed Camera Model")
        );
        assert_eq!(
            ascii_value(&exif, exif::Tag::DateTimeOriginal).as_deref(),
            Some("2026:01:02 03:04:05")
        );
        assert!(exif.get_field(exif::Tag::Copyright, In::PRIMARY).is_none());
        assert_eq!(
            exif.get_field(exif::Tag::FNumber, In::PRIMARY)
                .and_then(|field| match &field.value {
                    Value::Rational(value) => value.first().map(|value| (value.num, value.denom)),
                    _ => None,
                }),
            Some((28, 10))
        );
        assert_eq!(
            exif.get_field(exif::Tag::FocalLength, In::PRIMARY)
                .and_then(|field| match &field.value {
                    Value::Rational(value) => value.first().map(|value| (value.num, value.denom)),
                    _ => None,
                }),
            Some((50, 1))
        );
        assert_eq!(
            exif.get_field(exif::Tag::ExposureTime, In::PRIMARY)
                .and_then(|field| match &field.value {
                    Value::Rational(value) => value.first().map(|value| (value.num, value.denom)),
                    _ => None,
                }),
            Some((1, 125))
        );
        assert_eq!(
            exif.get_field(exif::Tag::PhotographicSensitivity, In::PRIMARY)
                .and_then(|field| field.value.get_uint(0)),
            Some(400)
        );
        assert_eq!(
            exif.get_field(exif::Tag::ImageWidth, In::PRIMARY)
                .and_then(|field| field.value.get_uint(0)),
            Some(640)
        );
        assert_eq!(
            exif.get_field(exif::Tag::BitsPerSample, In::PRIMARY)
                .map(|field| (0..3).map(|i| field.value.get_uint(i)).collect::<Vec<_>>()),
            Some(vec![Some(16), Some(16), Some(16)])
        );
        assert_eq!(
            exif.get_field(exif::Tag::Orientation, In::PRIMARY)
                .and_then(|field| field.value.get_uint(0)),
            Some(1)
        );
        assert!(
            exif.get_field(exif::Tag::GPSLatitude, In::PRIMARY)
                .is_some()
        );
    }

    #[test]
    fn jxl_exif_strips_entire_gps_ifd() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jpg");
        write_typed_jpeg_source(&source);

        let tiff = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 32,
                height: 24,
                bits_per_sample: 8,
                samples_per_pixel: 4,
            },
        )
        .unwrap();
        let exif = parse_tiff(tiff);

        assert!(
            [
                exif::Tag::GPSInfoIFDPointer,
                exif::Tag::GPSLatitude,
                exif::Tag::GPSLongitude,
                exif::Tag::GPSAltitude,
            ]
            .iter()
            .all(|tag| exif.get_field(*tag, In::PRIMARY).is_none())
        );
    }

    #[test]
    fn jxl_exif_applies_virtual_copy_sidecar_overrides() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jpg");
        let virtual_sidecar_path = temp.path().join("source.jpg.7.rrdata");
        write_typed_jpeg_source(&source);

        let mut virtual_sidecar = ImageMetadata::default();
        virtual_sidecar.exif_overrides = Some(HashMap::from([(
            "LensModel".to_string(),
            Some("Virtual Copy Lens".to_string()),
        )]));
        save_primary_metadata(&source, &ImageMetadata::default()).unwrap();
        fs::write(
            &virtual_sidecar_path,
            serde_json::to_vec_pretty(&virtual_sidecar).unwrap(),
        )
        .unwrap();

        let tiff = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            Some(&virtual_sidecar_path),
            true,
            RenderedImageMetadata {
                width: 32,
                height: 24,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .unwrap();
        let exif = parse_tiff(tiff);
        assert_eq!(
            ascii_value(&exif, exif::Tag::LensModel).as_deref(),
            Some("Virtual Copy Lens")
        );
    }

    #[test]
    fn invalid_explicit_sidecar_value_fails_jxl_metadata_build() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jpg");
        write_typed_jpeg_source(&source);

        let mut sidecar = ImageMetadata::default();
        sidecar.exif_overrides = Some(HashMap::from([(
            "FNumber".to_string(),
            Some("not-a-rational".to_string()),
        )]));
        save_primary_metadata(&source, &sidecar).unwrap();

        let error = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 32,
                height: 24,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .unwrap_err();
        assert!(error.contains("FNumber"));
    }

    #[test]
    fn jxl_metadata_allows_a_valid_source_without_exif() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("plain.jpg");
        DynamicImage::new_rgb8(2, 2).save(&source).unwrap();

        build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 2,
                height: 2,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .expect("a valid source with no EXIF should retain derived export metadata");
    }

    #[test]
    fn jxl_metadata_allows_supported_sources_without_exif_containers() {
        let temp = tempfile::tempdir().unwrap();
        let bmp = temp.path().join("plain.bmp");
        let gif = temp.path().join("plain.gif");
        DynamicImage::new_rgb8(2, 2)
            .save_with_format(&bmp, image::ImageFormat::Bmp)
            .unwrap();
        DynamicImage::new_rgb8(2, 2)
            .save_with_format(&gif, image::ImageFormat::Gif)
            .unwrap();

        let plain_jxl = temp.path().join("plain.jxl");
        let plain_jxl_bytes = jxl_encoder::LosslessConfig::new()
            .encode(&[0u8; 12], 2, 2, jxl_encoder::PixelLayout::Rgb8)
            .unwrap();
        fs::write(&plain_jxl, plain_jxl_bytes).unwrap();

        for source in [&bmp, &gif, &plain_jxl] {
            build_jxl_exif_tiff(
                source.to_str().unwrap(),
                None,
                true,
                RenderedImageMetadata {
                    width: 2,
                    height: 2,
                    bits_per_sample: 8,
                    samples_per_pixel: 3,
                },
            )
            .unwrap_or_else(|error| {
                panic!(
                    "source without a readable EXIF container should be accepted ({}): {error}",
                    source.display()
                )
            });
        }
    }

    #[test]
    fn jxl_source_exif_box_is_preserved_by_the_whitelist() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jxl");

        let mut source_metadata = Metadata::new();
        source_metadata.set_tag(ExifTag::Make("JXL Source Camera".to_string()));
        source_metadata.set_tag(ExifTag::FNumber(vec![rational(28, 10)]));
        let source_tiff = source_metadata.encode().unwrap();
        let encoder_metadata = jxl_encoder::ImageMetadata::new().with_exif(&source_tiff);
        let source_bytes = jxl_encoder::LosslessConfig::new()
            .encode_request(1, 1, jxl_encoder::PixelLayout::Rgb8)
            .with_metadata(&encoder_metadata)
            .encode(&[1u8, 2, 3])
            .unwrap();
        fs::write(&source, source_bytes).unwrap();

        let output_tiff = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 1,
                height: 1,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .unwrap();
        let output_exif = parse_tiff(output_tiff);
        assert_eq!(
            ascii_value(&output_exif, exif::Tag::Make).as_deref(),
            Some("JXL Source Camera")
        );
        assert!(
            output_exif
                .get_field(exif::Tag::FNumber, exif::In::PRIMARY)
                .is_some()
        );
    }

    #[test]
    fn malformed_jxl_source_exif_fails_metadata_build() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("malformed-exif.jxl");
        let encoder_metadata = jxl_encoder::ImageMetadata::new().with_exif(b"not a TIFF payload");
        let source_bytes = jxl_encoder::LosslessConfig::new()
            .encode_request(1, 1, jxl_encoder::PixelLayout::Rgb8)
            .with_metadata(&encoder_metadata)
            .encode(&[1u8, 2, 3])
            .unwrap();
        fs::write(&source, source_bytes).unwrap();

        let error = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 1,
                height: 1,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .unwrap_err();
        assert!(error.contains("Failed to parse source JXL EXIF"));

        let mut jpeg_output = Vec::new();
        DynamicImage::new_rgb8(1, 1)
            .write_to(&mut Cursor::new(&mut jpeg_output), image::ImageFormat::Jpeg)
            .unwrap();
        write_image_with_metadata(
            &mut jpeg_output,
            source.to_str().unwrap(),
            "jpeg",
            true,
            false,
            None,
            None,
        )
        .expect("JPEG must retain its existing best-effort metadata behavior");
    }

    #[test]
    fn malformed_source_exif_fails_jxl_metadata_build() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("malformed.jpg");
        fs::write(
            &source,
            [
                0xff, 0xd8, 0xff, 0xe1, 0x00, 0x10, b'E', b'x', b'i', b'f', 0x00, 0x00,
            ],
        )
        .unwrap();

        let error = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 2,
                height: 2,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .unwrap_err();
        assert!(error.contains("Failed to parse source EXIF"));
    }

    #[test]
    fn malformed_explicit_sidecar_fails_jxl_metadata_build() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.jpg");
        write_typed_jpeg_source(&source);
        fs::write(get_primary_sidecar_path(&source), b"{not valid json").unwrap();

        let error = build_jxl_exif_tiff(
            source.to_str().unwrap(),
            None,
            true,
            RenderedImageMetadata {
                width: 2,
                height: 2,
                bits_per_sample: 8,
                samples_per_pixel: 3,
            },
        )
        .unwrap_err();
        assert!(error.contains("Failed to parse JXL metadata sidecar"));
    }
}
