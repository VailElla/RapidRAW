use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use image::imageops::{self, FilterType};
use image::{
    DynamicImage, GenericImageView, GrayImage, Rgb, Rgb32FImage, RgbImage, Rgba, RgbaImage,
};
use ndarray::{Array, Array4, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::Emitter;
use tauri::Manager;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

const SAM3_VISION_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam3_vision.onnx?download=true";
const SAM3_VISION_FILENAME: &str = "sam3_vision.onnx";
const SAM3_VISION_SHA256: &str = "da4b6aca84712ec8cb99d775f514db2897de0f9a28a4bab828206d01dacefbe9";
const SAM3_TEXT_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam3_text.onnx?download=true";
const SAM3_TEXT_FILENAME: &str = "sam3_text.onnx";
const SAM3_TEXT_SHA256: &str = "ebd1b0576512e088a666c644d44b61f85bfe91537e99a025b1166c6b23085c1d";
const SAM3_DECODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam3_decoder.onnx?download=true";
const SAM3_DECODER_FILENAME: &str = "sam3_decoder.onnx";
const SAM3_DECODER_SHA256: &str =
    "0215dd748a5efed1a5af8dc894667cba47aad9d74ad949bfbfd11352bf654d03";
const SAM3_TOKENIZER_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/raw/main/sam3_tokenizer.json?download=true";
const SAM3_TOKENIZER_FILENAME: &str = "sam3_tokenizer.json";
const SAM3_INPUT_SIZE: u32 = 1008;

const U2NETP_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/u2net.onnx?download=true";
const U2NETP_FILENAME: &str = "u2net.onnx";
const U2NETP_INPUT_SIZE: u32 = 320;
const U2NETP_SHA256: &str = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491";

const SKYSEG_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/skyseg-u2net.onnx?download=true";
const SKYSEG_FILENAME: &str = "skyseg_u2net.onnx";
const SKYSEG_INPUT_SIZE: u32 = 320;
const SKYSEG_SHA256: &str = "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39";

const CLIP_MODEL_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_model.onnx?download=true";
const CLIP_MODEL_FILENAME: &str = "clip_model.onnx";
const CLIP_TOKENIZER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_tokenizer.json?download=true";
const CLIP_TOKENIZER_FILENAME: &str = "clip_tokenizer.json";
const CLIP_MODEL_SHA256: &str = "57879bb1c23cdeb350d23569dd251ed4b740a96d747c529e94a2bb8040ac5d00";

const DENOISE_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/nind_denoise_utnet_684.onnx?download=true";
const DENOISE_FILENAME: &str = "nind_denoise_utnet_684.onnx";
const DENOISE_SHA256: &str = "ee3586279d514df557ff3f7dec6df37fafc51ba5d3a3435b2cc9ac2d9017e7fe";

const LAMA_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/lama_fp16.onnx?download=true";
const LAMA_FILENAME: &str = "lama_fp16.onnx";
const LAMA_SHA256: &str = "2d6be6277c400d6f1b91819737f7c3da935e5c63d1b521d393be1196a2bfa82c";

const DEPTH_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/depth_anything_v2_vits.onnx?download=true";
const DEPTH_FILENAME: &str = "depth_anything_v2_vits.onnx";
const DEPTH_INPUT_SIZE: u32 = 518;
const DEPTH_SHA256: &str = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be";

pub struct AiModels {
    pub sam3_vision: Mutex<Session>,
    pub sam3_text: Mutex<Session>,
    pub sam3_decoder: Mutex<Session>,
    pub sam3_tokenizer: Tokenizer,
    pub u2netp: Mutex<Session>,
    pub sky_seg: Mutex<Session>,
    pub depth_anything: Mutex<Session>,
}

pub struct ClipModels {
    pub model: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

#[derive(Clone)]
pub struct ImageEmbeddings {
    pub path_hash: String,
    pub fpn_feat_0: Array<f32, IxDyn>,
    pub fpn_feat_1: Array<f32, IxDyn>,
    pub fpn_feat_2: Array<f32, IxDyn>,
    pub fpn_pos_2: Array<f32, IxDyn>,
    pub original_size: (u32, u32),
}

#[derive(Clone)]
pub struct CachedDepthMap {
    pub path_hash: String,
    pub depth_image: GrayImage,
    pub original_size: (u32, u32),
}

pub struct AiState {
    pub models: Option<Arc<AiModels>>,
    pub denoise_model: Option<Arc<Mutex<Session>>>,
    pub clip_models: Option<Arc<ClipModels>>,
    pub lama_model: Option<Arc<Mutex<Session>>>,
    pub embeddings: Option<ImageEmbeddings>,
    pub depth_map: Option<CachedDepthMap>,
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf> {
    let models_dir = app_handle.path().app_data_dir()?.join("models");
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }
    Ok(models_dir)
}

async fn download_model(url: &str, dest: &Path) -> Result<()> {
    let response = reqwest::get(url).await?;
    let mut file = fs::File::create(dest)?;
    let mut content = Cursor::new(response.bytes().await?);
    std::io::copy(&mut content, &mut file)?;
    Ok(())
}

fn verify_sha256(path: &Path, expected_hash: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    if expected_hash.is_empty() {
        return Ok(true);
    }
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let hash = hasher.finalize();
    let hex_hash = hex::encode(hash);
    Ok(hex_hash == expected_hash)
}

async fn download_and_verify_model(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    filename: &str,
    url: &str,
    expected_hash: &str,
    model_name: &str,
) -> Result<()> {
    let dest_path = models_dir.join(filename);
    let is_valid = verify_sha256(&dest_path, expected_hash)?;

    if !is_valid {
        if dest_path.exists() {
            println!("Model {} has incorrect hash. Re-downloading.", model_name);
            fs::remove_file(&dest_path)?;
        }
        let _ = app_handle.emit("ai-model-download-start", model_name);
        download_model(url, &dest_path).await?;
        let _ = app_handle.emit("ai-model-download-finish", model_name);

        if !verify_sha256(&dest_path, expected_hash)? {
            return Err(anyhow::anyhow!(
                "Failed to verify model {} after download. Hash mismatch.",
                model_name
            ));
        }
    }
    Ok(())
}

pub async fn get_or_init_ai_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<AiModels>> {
    if let Some(models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.models.clone())
    {
        return Ok(models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.models.clone())
    {
        return Ok(models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(
        app_handle,
        &models_dir,
        SAM3_VISION_FILENAME,
        SAM3_VISION_URL,
        SAM3_VISION_SHA256,
        "SAM3 Vision",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        SAM3_TEXT_FILENAME,
        SAM3_TEXT_URL,
        SAM3_TEXT_SHA256,
        "SAM3 Text",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        SAM3_DECODER_FILENAME,
        SAM3_DECODER_URL,
        SAM3_DECODER_SHA256,
        "SAM3 Decoder",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        U2NETP_FILENAME,
        U2NETP_URL,
        U2NETP_SHA256,
        "Foreground Model",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        SKYSEG_FILENAME,
        SKYSEG_URL,
        SKYSEG_SHA256,
        "Sky Model",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DEPTH_FILENAME,
        DEPTH_URL,
        DEPTH_SHA256,
        "Depth Model",
    )
    .await?;

    let tokenizer_path = models_dir.join(SAM3_TOKENIZER_FILENAME);
    if !tokenizer_path.exists() {
        let _ = app_handle.emit("ai-model-download-start", "SAM3 Tokenizer");
        download_model(SAM3_TOKENIZER_URL, &tokenizer_path).await?;
        let _ = app_handle.emit("ai-model-download-finish", "SAM3 Tokenizer");
    }

    let _ = ort::init().with_name("AI").commit();

    let sam3_vision_path = models_dir.join(SAM3_VISION_FILENAME);
    let sam3_text_path = models_dir.join(SAM3_TEXT_FILENAME);
    let sam3_decoder_path = models_dir.join(SAM3_DECODER_FILENAME);
    let u2netp_path = models_dir.join(U2NETP_FILENAME);
    let sky_seg_path = models_dir.join(SKYSEG_FILENAME);
    let depth_path = models_dir.join(DEPTH_FILENAME);

    let sam3_vision = Session::builder()?.commit_from_file(sam3_vision_path)?;
    let sam3_text = Session::builder()?.commit_from_file(sam3_text_path)?;
    let sam3_decoder = Session::builder()?.commit_from_file(sam3_decoder_path)?;
    let sam3_tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let u2netp = Session::builder()?.commit_from_file(u2netp_path)?;
    let sky_seg = Session::builder()?.commit_from_file(sky_seg_path)?;
    let depth_anything = Session::builder()?.commit_from_file(depth_path)?;

    crate::register_exit_handler();

    let models = Arc::new(AiModels {
        sam3_vision: Mutex::new(sam3_vision),
        sam3_text: Mutex::new(sam3_text),
        sam3_decoder: Mutex::new(sam3_decoder),
        sam3_tokenizer,
        u2netp: Mutex::new(u2netp),
        sky_seg: Mutex::new(sky_seg),
        depth_anything: Mutex::new(depth_anything),
    });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.models = Some(models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: Some(models.clone()),
            denoise_model: None,
            clip_models: None,
            lama_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(models)
}

pub async fn get_or_init_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DENOISE_FILENAME,
        DENOISE_URL,
        DENOISE_SHA256,
        "NIND Denoise Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let model_path = models_dir.join(DENOISE_FILENAME);
    let session = Session::builder()?.commit_from_file(model_path)?;
    let denoise_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.denoise_model = Some(denoise_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: Some(denoise_model.clone()),
            clip_models: None,
            lama_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(denoise_model)
}

pub async fn get_or_init_clip_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<ClipModels>> {
    if let Some(clip_models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.clip_models.clone())
    {
        return Ok(clip_models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(clip_models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.clip_models.clone())
    {
        return Ok(clip_models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(
        app_handle,
        &models_dir,
        CLIP_MODEL_FILENAME,
        CLIP_MODEL_URL,
        CLIP_MODEL_SHA256,
        "CLIP Model",
    )
    .await?;

    let clip_tokenizer_path = models_dir.join(CLIP_TOKENIZER_FILENAME);
    if !clip_tokenizer_path.exists() {
        let _ = app_handle.emit("ai-model-download-start", "CLIP Tokenizer");
        download_model(CLIP_TOKENIZER_URL, &clip_tokenizer_path).await?;
        let _ = app_handle.emit("ai-model-download-finish", "CLIP Tokenizer");
    }

    let _ = ort::init().with_name("AI-Tagging").commit();
    let clip_model_path = models_dir.join(CLIP_MODEL_FILENAME);
    let model = Mutex::new(Session::builder()?.commit_from_file(clip_model_path)?);
    let tokenizer =
        Tokenizer::from_file(clip_tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    crate::register_exit_handler();

    let clip_models = Arc::new(ClipModels { model, tokenizer });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.clip_models = Some(clip_models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            clip_models: Some(clip_models.clone()),
            lama_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(clip_models)
}

pub async fn get_or_init_lama_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(lama_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_model.clone())
    {
        return Ok(lama_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(lama_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_model.clone())
    {
        return Ok(lama_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        LAMA_FILENAME,
        LAMA_URL,
        LAMA_SHA256,
        "Inpainting Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Inpainting").commit();
    let model_path = models_dir.join(LAMA_FILENAME);
    let session = Session::builder()?.commit_from_file(model_path)?;
    let lama_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.lama_model = Some(lama_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            clip_models: None,
            lama_model: Some(lama_model.clone()),
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(lama_model)
}

fn box_filter_f32(data: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut temp = vec![0.0; w * h];
    let mut out = vec![0.0; w * h];

    for y in 0..h {
        let offset = y * w;
        for x in 0..w {
            let min_x = x.saturating_sub(r);
            let max_x = (x + r).min(w - 1);
            let mut sum = 0.0;
            for xi in min_x..=max_x {
                sum += data[offset + xi];
            }
            temp[offset + x] = sum / (max_x - min_x + 1) as f32;
        }
    }
    for x in 0..w {
        for y in 0..h {
            let min_y = y.saturating_sub(r);
            let max_y = (y + r).min(h - 1);
            let mut sum = 0.0;
            for yi in min_y..=max_y {
                sum += temp[yi * w + x];
            }
            out[y * w + x] = sum / (max_y - min_y + 1) as f32;
        }
    }
    out
}

fn resize_f32_bilinear(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let mut dst = vec![0.0; dst_w * dst_h];
    let x_ratio = (src_w as f32 - 1.0) / (dst_w as f32 - 1.0).max(1.0);
    let y_ratio = (src_h as f32 - 1.0) / (dst_h as f32 - 1.0).max(1.0);

    for y in 0..dst_h {
        let src_y = y as f32 * y_ratio;
        let y_lo = src_y.floor() as usize;
        let y_hi = (y_lo + 1).min(src_h - 1);
        let y_weight = src_y - y_lo as f32;

        let row_lo = y_lo * src_w;
        let row_hi = y_hi * src_w;
        let dst_row = y * dst_w;

        for x in 0..dst_w {
            let src_x = x as f32 * x_ratio;
            let x_lo = src_x.floor() as usize;
            let x_hi = (x_lo + 1).min(src_w - 1);
            let x_weight = src_x - x_lo as f32;

            let tl = src[row_lo + x_lo];
            let tr = src[row_lo + x_hi];
            let bl = src[row_hi + x_lo];
            let br = src[row_hi + x_hi];

            let top = tl + (tr - tl) * x_weight;
            let bottom = bl + (br - bl) * x_weight;
            dst[dst_row + x] = top + (bottom - top) * y_weight;
        }
    }
    dst
}

pub fn fast_color_guided_filter(
    high_res_guide: &RgbImage,
    low_res_mask: &GrayImage,
    r: usize,
    eps: f32,
    sharpen_k: f32,
    midpoint: f32,
) -> GrayImage {
    let (w_high, h_high) = high_res_guide.dimensions();
    let (w_low, h_low) = low_res_mask.dimensions();

    let low_res_guide = imageops::resize(high_res_guide, w_low, h_low, FilterType::Triangle);

    let wl = w_low as usize;
    let hl = h_low as usize;
    let num_pixels = wl * hl;

    let mut ir = vec![0.0; num_pixels];
    let mut ig = vec![0.0; num_pixels];
    let mut ib = vec![0.0; num_pixels];
    let mut p = vec![0.0; num_pixels];

    for (i, (pix_g, pix_m)) in low_res_guide
        .pixels()
        .zip(low_res_mask.pixels())
        .enumerate()
    {
        ir[i] = pix_g[0] as f32 / 255.0;
        ig[i] = pix_g[1] as f32 / 255.0;
        ib[i] = pix_g[2] as f32 / 255.0;
        p[i] = pix_m[0] as f32 / 255.0;
    }

    let p_high = resize_f32_bilinear(&p, wl, hl, w_high as usize, h_high as usize);

    let mean_ir = box_filter_f32(&ir, wl, hl, r);
    let mean_ig = box_filter_f32(&ig, wl, hl, r);
    let mean_ib = box_filter_f32(&ib, wl, hl, r);
    let mean_p = box_filter_f32(&p, wl, hl, r);

    let mut var_rr = vec![0.0; num_pixels];
    let mut var_rg = vec![0.0; num_pixels];
    let mut var_rb = vec![0.0; num_pixels];
    let mut var_gg = vec![0.0; num_pixels];
    let mut var_gb = vec![0.0; num_pixels];
    let mut var_bb = vec![0.0; num_pixels];

    let mut cov_rp = vec![0.0; num_pixels];
    let mut cov_gp = vec![0.0; num_pixels];
    let mut cov_bp = vec![0.0; num_pixels];

    for i in 0..num_pixels {
        var_rr[i] = ir[i] * ir[i];
        var_rg[i] = ir[i] * ig[i];
        var_rb[i] = ir[i] * ib[i];
        var_gg[i] = ig[i] * ig[i];
        var_gb[i] = ig[i] * ib[i];
        var_bb[i] = ib[i] * ib[i];

        cov_rp[i] = ir[i] * p[i];
        cov_gp[i] = ig[i] * p[i];
        cov_bp[i] = ib[i] * p[i];
    }

    let mean_rr = box_filter_f32(&var_rr, wl, hl, r);
    let mean_rg = box_filter_f32(&var_rg, wl, hl, r);
    let mean_rb = box_filter_f32(&var_rb, wl, hl, r);
    let mean_gg = box_filter_f32(&var_gg, wl, hl, r);
    let mean_gb = box_filter_f32(&var_gb, wl, hl, r);
    let mean_bb = box_filter_f32(&var_bb, wl, hl, r);

    let mean_rp = box_filter_f32(&cov_rp, wl, hl, r);
    let mean_gp = box_filter_f32(&cov_gp, wl, hl, r);
    let mean_bp = box_filter_f32(&cov_bp, wl, hl, r);

    let mut a_r = vec![0.0; num_pixels];
    let mut a_g = vec![0.0; num_pixels];
    let mut a_b = vec![0.0; num_pixels];
    let mut b = vec![0.0; num_pixels];

    for i in 0..num_pixels {
        let s00 = mean_rr[i] - mean_ir[i] * mean_ir[i] + eps;
        let s01 = mean_rg[i] - mean_ir[i] * mean_ig[i];
        let s02 = mean_rb[i] - mean_ir[i] * mean_ib[i];
        let s11 = mean_gg[i] - mean_ig[i] * mean_ig[i] + eps;
        let s12 = mean_gb[i] - mean_ig[i] * mean_ib[i];
        let s22 = mean_bb[i] - mean_ib[i] * mean_ib[i] + eps;

        let crp = mean_rp[i] - mean_ir[i] * mean_p[i];
        let cgp = mean_gp[i] - mean_ig[i] * mean_p[i];
        let cbp = mean_bp[i] - mean_ib[i] * mean_p[i];

        let det = s00 * (s11 * s22 - s12 * s12) - s01 * (s01 * s22 - s12 * s02)
            + s02 * (s01 * s12 - s11 * s02);

        if det.abs() > 1e-10 {
            let inv00 = (s11 * s22 - s12 * s12) / det;
            let inv01 = (s02 * s12 - s01 * s22) / det;
            let inv02 = (s01 * s12 - s11 * s02) / det;
            let inv11 = (s00 * s22 - s02 * s02) / det;
            let inv12 = (s01 * s02 - s00 * s12) / det;
            let inv22 = (s00 * s11 - s01 * s01) / det;

            a_r[i] = inv00 * crp + inv01 * cgp + inv02 * cbp;
            a_g[i] = inv01 * crp + inv11 * cgp + inv12 * cbp;
            a_b[i] = inv02 * crp + inv12 * cgp + inv22 * cbp;
        } else {
            a_r[i] = 0.0;
            a_g[i] = 0.0;
            a_b[i] = 0.0;
        }

        b[i] = mean_p[i] - (a_r[i] * mean_ir[i] + a_g[i] * mean_ig[i] + a_b[i] * mean_ib[i]);
    }

    let wh = w_high as usize;
    let hh = h_high as usize;

    let a_r_high = resize_f32_bilinear(&a_r, wl, hl, wh, hh);
    let a_g_high = resize_f32_bilinear(&a_g, wl, hl, wh, hh);
    let a_b_high = resize_f32_bilinear(&a_b, wl, hl, wh, hh);
    let b_high = resize_f32_bilinear(&b, wl, hl, wh, hh);

    let mut out_mask = GrayImage::new(w_high, h_high);

    let sig = |x: f32| -> f32 { 1.0 / (1.0 + (-sharpen_k * (x - midpoint)).exp()) };
    let v_min = if sharpen_k > 0.0 { sig(0.0) } else { 0.0 };
    let v_max = if sharpen_k > 0.0 { sig(1.0) } else { 1.0 };
    let v_range = (v_max - v_min).max(1e-6);

    for y in 0..hh {
        let y_guide = y as u32;
        for x in 0..wh {
            let idx = y * wh + x;
            let p_rgb = high_res_guide.get_pixel(x as u32, y_guide);
            let ir_h = p_rgb[0] as f32 / 255.0;
            let ig_h = p_rgb[1] as f32 / 255.0;
            let ib_h = p_rgb[2] as f32 / 255.0;

            let q =
                a_r_high[idx] * ir_h + a_g_high[idx] * ig_h + a_b_high[idx] * ib_h + b_high[idx];
            let q_clamped = q.clamp(0.0, 1.0);

            let q_sharp = if sharpen_k > 0.0 {
                (sig(q_clamped) - v_min) / v_range
            } else {
                q_clamped
            };

            let envelope = p_high[idx].clamp(0.0, 1.0);

            let q_final = q_sharp * envelope;
            let q_u8 = (q_final * 255.0).clamp(0.0, 255.0) as u8;
            out_mask.put_pixel(x as u32, y_guide, image::Luma([q_u8]));
        }
    }

    out_mask
}

#[derive(Clone, Copy)]
struct TileParams {
    cs: usize,
    ucs: usize,
    overlap: usize,
    pad: usize,
}

impl TileParams {
    const fn new(cs: usize, ucs: usize, overlap: usize) -> Self {
        Self {
            cs,
            ucs,
            overlap,
            pad: (cs - ucs) / 2,
        }
    }
}

const TILE_BALANCED: TileParams = TileParams::new(504, 480, 6);
const TILE_FASTER: TileParams = TileParams::new(504, 504, 0);
const TILE_HIGHER_QUALITY: TileParams = TileParams::new(504, 448, 12);

fn select_tile_params(quality_0_1: f32) -> TileParams {
    let q = quality_0_1.clamp(0.0, 1.0);
    if q <= 0.25 {
        TILE_FASTER
    } else if q >= 0.75 {
        TILE_HIGHER_QUALITY
    } else {
        TILE_BALANCED
    }
}

#[inline]
fn mirror_coord(c: i32, size: i32) -> i32 {
    if c < 0 {
        (-c).min(size - 1)
    } else if c >= size {
        (2 * size - 1 - c).max(0)
    } else {
        c
    }
}

fn extract_tile_mirror(img: &Rgb32FImage, x0: i32, y0: i32, cs: usize) -> Array4<f32> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut arr = Array4::zeros((1, 3, cs, cs));
    for dy in 0..cs as i32 {
        for dx in 0..cs as i32 {
            let sx = mirror_coord(x0 + dx, w);
            let sy = mirror_coord(y0 + dy, h);
            let px = img.get_pixel(sx as u32, sy as u32);
            arr[[0, 0, dy as usize, dx as usize]] = px[0];
            arr[[0, 1, dy as usize, dx as usize]] = px[1];
            arr[[0, 2, dy as usize, dx as usize]] = px[2];
        }
    }
    arr
}

struct SeamlessBlend {
    ud0: usize,
    ud1: usize,
    ud2: usize,
    ud3: usize,
    absx0: usize,
    absy0: usize,
    fswidth: usize,
    fsheight: usize,
    overlap: usize,
}

fn apply_seamless(tile: &mut Array4<f32>, blend: &SeamlessBlend) {
    let SeamlessBlend {
        ud0,
        ud1,
        ud2,
        ud3,
        absx0,
        absy0,
        fswidth,
        fsheight,
        overlap,
    } = *blend;
    let ol = overlap;
    if absx0 > 0 {
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in ud0..(ud0 + ol).min(ud2) {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 > 0 {
        for c in 0..3 {
            for y in ud1..(ud1 + ol).min(ud3) {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absx0 + (ud2 - ud0) < fswidth && ol > 0 {
        let right_start = (ud2 as i32 - ol as i32).max(ud0 as i32) as usize;
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in right_start..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 + (ud3 - ud1) < fsheight && ol > 0 {
        let bottom_start = (ud3 as i32 - ol as i32).max(ud1 as i32) as usize;
        for c in 0..3 {
            for y in bottom_start..ud3 {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
}

fn run_native_denoise(
    img: &Rgb32FImage,
    session: &Mutex<Session>,
    accumulator: &mut [f32],
    width: usize,
    height: usize,
    app_handle: &tauri::AppHandle,
    params: TileParams,
) -> Result<()> {
    let w = width as i32;
    let h = height as i32;
    let step = params.ucs.saturating_sub(params.overlap).max(1);
    let iperhl = (width.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let ipervl = (height.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let total = (iperhl + 1) * (ipervl + 1);

    for i in 0..total {
        let yi = i / (iperhl + 1);
        let xi = i % (iperhl + 1);
        let x0 =
            params.ucs as i32 * xi as i32 - params.overlap as i32 * xi as i32 - params.pad as i32;
        let y0 =
            params.ucs as i32 * yi as i32 - params.overlap as i32 * yi as i32 - params.pad as i32;

        if i % 10 == 0 {
            let pct = (i as f32 / total as f32) * 100.0;
            let _ = app_handle.emit("denoise-progress", format!("Denoising… {:.0}%", pct));
        }

        let crop = extract_tile_mirror(img, x0, y0, params.cs);
        let input_values = crop.as_standard_layout().to_owned();
        let t_input = Tensor::from_array(input_values)?;

        let out = {
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![t_input])?;
            let arr = outputs[0].try_extract_array::<f32>()?.to_owned();
            arr.into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))?
        };

        let x1pad = (0i32).max(x0 + params.cs as i32 - w) as usize;
        let y1pad = (0i32).max(y0 + params.cs as i32 - h) as usize;
        let ud0 = params.pad;
        let ud1 = params.pad;
        let ud2 = params.cs - params.pad.max(x1pad);
        let ud3 = params.cs - params.pad.max(y1pad);
        let absx0 = (x0 + params.pad as i32).max(0) as usize;
        let absy0 = (y0 + params.pad as i32).max(0) as usize;

        let mut tile = out;
        apply_seamless(
            &mut tile,
            &SeamlessBlend {
                ud0,
                ud1,
                ud2,
                ud3,
                absx0,
                absy0,
                fswidth: width,
                fsheight: height,
                overlap: params.overlap,
            },
        );

        for cy in 0..(ud3 - ud1) {
            for cx in 0..(ud2 - ud0) {
                let gx = absx0 + cx;
                let gy = absy0 + cy;
                if gx < width && gy < height {
                    let base = (gy * width + gx) * 3;
                    accumulator[base] += tile[[0, 0, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 1] += tile[[0, 1, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 2] += tile[[0, 2, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(())
}

fn accumulator_to_rgb32f(acc: &[f32], width: u32, height: u32) -> Rgb32FImage {
    let mut out = Rgb32FImage::new(width, height);
    for (i, p) in out.pixels_mut().enumerate() {
        let i3 = i * 3;
        *p = Rgb([
            acc[i3].clamp(0.0, 1.0),
            acc[i3 + 1].clamp(0.0, 1.0),
            acc[i3 + 2].clamp(0.0, 1.0),
        ]);
    }
    out
}

pub fn run_ai_denoise(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    session: &Mutex<Session>,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage> {
    let (width, height) = rgb_img.dimensions();
    let params = select_tile_params(intensity);

    let _ = app_handle.emit("denoise-progress", "Denoising (AI NIND)...");
    let mut accumulator = vec![0.0f32; width as usize * height as usize * 3];
    run_native_denoise(
        rgb_img,
        session,
        &mut accumulator,
        width as usize,
        height as usize,
        app_handle,
        params,
    )?;

    let out_img_buffer = accumulator_to_rgb32f(&accumulator, width, height);
    Ok(DynamicImage::ImageRgb32F(out_img_buffer))
}

pub fn run_lama_inpainting(
    image: &DynamicImage,
    mask: &GrayImage,
    lama_session: &Mutex<Session>,
) -> Result<RgbaImage> {
    let (w, h) = image.dimensions();

    let (mut min_x, mut min_y) = (w, h);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut has_mask = false;

    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 0 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            has_mask = true;
        }
    }

    if !has_mask {
        return Ok(image.to_rgba8());
    }

    let mask_w = max_x - min_x + 1;
    let mask_h = max_y - min_y + 1;

    let pad_x = 128.max((mask_w as f32 * 1.5) as u32);
    let pad_y = 128.max((mask_h as f32 * 1.5) as u32);

    let x0 = min_x.saturating_sub(pad_x);
    let y0 = min_y.saturating_sub(pad_y);
    let x1 = (max_x + pad_x).min(w.saturating_sub(1));
    let y1 = (max_y + pad_y).min(h.saturating_sub(1));

    let crop_w = x1 - x0 + 1;
    let crop_h = y1 - y0 + 1;

    let rgba = image.to_rgba8();

    let cropped_img = imageops::crop_imm(&rgba, x0, y0, crop_w, crop_h).to_image();
    let cropped_mask = imageops::crop_imm(mask, x0, y0, crop_w, crop_h).to_image();

    let max_dim_limit: u32 = 768;
    let needs_downscale = crop_w > max_dim_limit || crop_h > max_dim_limit;

    let (fw, fh, inf_img, inf_mask) = if needs_downscale {
        let scale = max_dim_limit as f32 / crop_w.max(crop_h) as f32;

        let scaled_w = (crop_w as f32 * scale).round().max(1.0) as u32;
        let scaled_h = (crop_h as f32 * scale).round().max(1.0) as u32;

        (
            scaled_w,
            scaled_h,
            imageops::resize(&cropped_img, scaled_w, scaled_h, FilterType::Lanczos3),
            imageops::resize(&cropped_mask, scaled_w, scaled_h, FilterType::Triangle),
        )
    } else {
        (crop_w, crop_h, cropped_img.clone(), cropped_mask.clone())
    };

    let align = 64u32;
    let mut tensor_dim = fw.max(fh);
    if tensor_dim % align != 0 {
        tensor_dim += align - (tensor_dim % align);
    }
    let tensor_dim = tensor_dim.max(align) as usize;

    let mut img_tensor = Array::<f32, _>::zeros((1, 3, tensor_dim, tensor_dim));
    let mut msk_tensor = Array::<f32, _>::zeros((1, 1, tensor_dim, tensor_dim));

    for y in 0..tensor_dim {
        for x in 0..tensor_dim {
            let sx = (x as u32).min(fw.saturating_sub(1));
            let sy = (y as u32).min(fh.saturating_sub(1));

            let p = inf_img.get_pixel(sx, sy);
            let m = inf_mask.get_pixel(sx, sy)[0];

            img_tensor[[0, 0, y, x]] = p[0] as f32 / 255.0;
            img_tensor[[0, 1, y, x]] = p[1] as f32 / 255.0;
            img_tensor[[0, 2, y, x]] = p[2] as f32 / 255.0;
            msk_tensor[[0, 0, y, x]] = if m > 0 { 1.0 } else { 0.0 };
        }
    }

    let t_img = Tensor::from_array(img_tensor.into_dyn().as_standard_layout().into_owned())?;
    let t_msk = Tensor::from_array(msk_tensor.into_dyn().as_standard_layout().into_owned())?;

    let output_tensor = {
        let mut session = lama_session.lock().unwrap();
        let outputs = session.run(ort::inputs!["image" => t_img, "mask" => t_msk])?;
        outputs[0].try_extract_array::<f32>()?.to_owned()
    };

    let mut result_inf = RgbaImage::new(fw, fh);
    for y in 0..fh {
        for x in 0..fw {
            let r = output_tensor[[0, 0, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            let g = output_tensor[[0, 1, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            let b = output_tensor[[0, 2, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            result_inf.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    let result_crop = if needs_downscale {
        imageops::resize(&result_inf, crop_w, crop_h, FilterType::Lanczos3)
    } else {
        result_inf
    };

    let mut final_image = image.to_rgba8();

    for y in 0..crop_h {
        for x in 0..crop_w {
            let m = cropped_mask.get_pixel(x, y)[0];
            if m > 0 {
                let alpha = m as f32 / 255.0;
                let p = result_crop.get_pixel(x, y);
                let gx = x0 + x;
                let gy = y0 + y;
                let orig = final_image.get_pixel(gx, gy);

                let r = (p[0] as f32 * alpha + orig[0] as f32 * (1.0 - alpha)) as u8;
                let g = (p[1] as f32 * alpha + orig[1] as f32 * (1.0 - alpha)) as u8;
                let b = (p[2] as f32 * alpha + orig[2] as f32 * (1.0 - alpha)) as u8;

                final_image.put_pixel(gx, gy, Rgba([r, g, b, 255]));
            }
        }
    }

    Ok(final_image)
}

pub fn generate_image_embeddings(
    image: &DynamicImage,
    sam3_vision: &Mutex<Session>,
) -> Result<ImageEmbeddings> {
    let (orig_width, orig_height) = image.dimensions();

    let long_side = orig_width.max(orig_height) as f32;
    let scale = SAM3_INPUT_SIZE as f32 / long_side;
    let new_width = (orig_width as f32 * scale).round() as u32;
    let new_height = (orig_height as f32 * scale).round() as u32;

    let resized_image = image.resize_exact(new_width, new_height, FilterType::Triangle);
    let rgb_image = resized_image.into_rgb8();
    let (actual_width, actual_height) = rgb_image.dimensions();
    let raw_pixels = rgb_image.as_raw();

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, SAM3_INPUT_SIZE as usize, SAM3_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let w_usize = actual_width as usize;
    for y in 0..(actual_height as usize) {
        for x in 0..w_usize {
            let idx = (y * w_usize + x) * 3;
            input_tensor[[0, 0, y, x]] = ((raw_pixels[idx] as f32 / 255.0) - mean[0]) / std[0];
            input_tensor[[0, 1, y, x]] = ((raw_pixels[idx + 1] as f32 / 255.0) - mean[1]) / std[1];
            input_tensor[[0, 2, y, x]] = ((raw_pixels[idx + 2] as f32 / 255.0) - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let input_values = input_tensor_dyn.as_standard_layout();
    let input_tensor_ort = Tensor::from_array(input_values.into_owned())?;

    let mut session = sam3_vision.lock().unwrap();
    let outputs = session.run(ort::inputs![input_tensor_ort])?;

    Ok(ImageEmbeddings {
        path_hash: "".to_string(),
        fpn_feat_0: outputs[0].try_extract_array::<f32>()?.to_owned().into_dyn(),
        fpn_feat_1: outputs[1].try_extract_array::<f32>()?.to_owned().into_dyn(),
        fpn_feat_2: outputs[2].try_extract_array::<f32>()?.to_owned().into_dyn(),
        fpn_pos_2: outputs[3].try_extract_array::<f32>()?.to_owned().into_dyn(),
        original_size: (orig_width, orig_height),
    })
}

pub fn run_sam3_decoder(
    sam3_text: &Mutex<Session>,
    decoder: &Mutex<Session>,
    tokenizer: &Tokenizer,
    embeddings: &ImageEmbeddings,
    text_prompt: &str,
    use_box: bool,
    start_point: (f64, f64),
    end_point: (f64, f64),
    high_res_guide: &DynamicImage,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = embeddings.original_size;

    let clean_prompt = text_prompt.trim();
    let dynamic_prompt = if clean_prompt.is_empty() {
        "visual object"
    } else {
        clean_prompt
    };

    let encoding = tokenizer
        .encode(dynamic_prompt, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut ids = encoding.get_ids().to_vec();
    let mut mask = encoding.get_attention_mask().to_vec();

    ids.resize(32, 0);
    mask.resize(32, 0);

    let ids_arr = Array::from_shape_vec((1, 32), ids.into_iter().map(|i| i as i64).collect())?;
    let mask_arr = Array::from_shape_vec((1, 32), mask.into_iter().map(|m| m as i64).collect())?;

    let t_ids = Tensor::from_array(ids_arr.into_dyn().as_standard_layout().into_owned())?;
    let t_mask = Tensor::from_array(mask_arr.into_dyn().as_standard_layout().into_owned())?;

    let (text_features, text_out_mask) = {
        let mut session = sam3_text.lock().unwrap();
        let text_outputs = session.run(ort::inputs![t_ids, t_mask])?;

        let f = if let Ok(feat) = text_outputs[0].try_extract_array::<f32>() {
            feat.to_owned().into_dyn()
        } else {
            anyhow::bail!("Failed to extract text_features");
        };

        let m = if let Ok(m_bool) = text_outputs[1].try_extract_array::<bool>() {
            m_bool.to_owned().into_dyn()
        } else if let Ok(m_f32) = text_outputs[1].try_extract_array::<f32>() {
            m_f32.mapv(|x| x != 0.0).into_dyn()
        } else if let Ok(m_i64) = text_outputs[1].try_extract_array::<i64>() {
            m_i64.mapv(|x| x != 0).into_dyn()
        } else {
            anyhow::bail!("Failed to extract text_mask");
        };

        (f, m)
    };

    let (boxes, labels) = if use_box {
        let rx1 = start_point.0.min(end_point.0).clamp(0.0, orig_width as f64);
        let ry1 = start_point
            .1
            .min(end_point.1)
            .clamp(0.0, orig_height as f64);
        let rx2 = start_point.0.max(end_point.0).clamp(0.0, orig_width as f64);
        let ry2 = start_point
            .1
            .max(end_point.1)
            .clamp(0.0, orig_height as f64);

        let rw = rx2 - rx1;
        let rh = ry2 - ry1;

        let rcx = rx1 + rw / 2.0;
        let rcy = ry1 + rh / 2.0;

        let cx = (rcx / orig_width as f64) as f32;
        let cy = (rcy / orig_height as f64) as f32;
        let nw = (rw / orig_width as f64) as f32;
        let nh = (rh / orig_height as f64) as f32;

        (
            Array::from_shape_vec((1, 1, 4), vec![cx, cy, nw, nh])?,
            Array::from_shape_vec((1, 1), vec![1_i64])?,
        )
    } else {
        (
            Array::from_shape_vec((1, 1, 4), vec![0.0, 0.0, 0.0, 0.0])?,
            Array::from_shape_vec((1, 1), vec![-10_i64])?,
        )
    };

    let t_fpn_0 = Tensor::from_array(
        embeddings
            .fpn_feat_0
            .clone()
            .as_standard_layout()
            .into_owned(),
    )?;
    let t_fpn_1 = Tensor::from_array(
        embeddings
            .fpn_feat_1
            .clone()
            .as_standard_layout()
            .into_owned(),
    )?;
    let t_fpn_2 = Tensor::from_array(
        embeddings
            .fpn_feat_2
            .clone()
            .as_standard_layout()
            .into_owned(),
    )?;
    let t_fpn_pos_2 = Tensor::from_array(
        embeddings
            .fpn_pos_2
            .clone()
            .as_standard_layout()
            .into_owned(),
    )?;
    let t_text_feat = Tensor::from_array(text_features.as_standard_layout().into_owned())?;
    let t_text_mask = Tensor::from_array(text_out_mask.as_standard_layout().into_owned())?;
    let t_boxes = Tensor::from_array(boxes.into_dyn().as_standard_layout().into_owned())?;
    let t_labels = Tensor::from_array(labels.into_dyn().as_standard_layout().into_owned())?;

    let (masks, logits, presence) = {
        let mut session = decoder.lock().unwrap();
        let decoder_outputs = session.run(ort::inputs![
            t_fpn_0,
            t_fpn_1,
            t_fpn_2,
            t_fpn_pos_2,
            t_text_feat,
            t_text_mask,
            t_boxes,
            t_labels
        ])?;

        let m = decoder_outputs[0].try_extract_array::<f32>()?.to_owned();
        let l = decoder_outputs[2].try_extract_array::<f32>()?.to_owned();
        let p = decoder_outputs[3].try_extract_array::<f32>()?.to_owned();

        (m, l, p)
    };

    let logits_slice = logits.as_slice().unwrap();
    let presence_slice = presence.as_slice().unwrap();

    let mut best_idx = 0;
    let mut best_score = f32::MIN;

    let p_score = 1.0 / (1.0 + (-presence_slice[0]).exp());
    for i in 0..logits_slice.len() {
        let score = (1.0 / (1.0 + (-logits_slice[i]).exp())) * p_score;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let mask_dims = masks.shape();
    let h = mask_dims[2] as usize;
    let w = mask_dims[3] as usize;

    let masks_4d = masks
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| anyhow::anyhow!("Expected 4D mask array: {}", e))?;

    let mask_view = masks_4d.slice(ndarray::s![0, best_idx, .., ..]);
    let mut mask_data = Vec::with_capacity(h * w);
    for val in mask_view.iter() {
        let prob = 1.0 / (1.0 + (-val).exp());
        mask_data.push((prob * 255.0).clamp(0.0, 255.0) as u8);
    }

    let mask_img = GrayImage::from_raw(w as u32, h as u32, mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask"))?;

    let long_side_u32 = orig_width.max(orig_height);
    let valid_w = ((orig_width as f64 / long_side_u32 as f64) * w as f64).round() as u32;
    let valid_h = ((orig_height as f64 / long_side_u32 as f64) * h as f64).round() as u32;

    let low_res_cropped =
        imageops::crop_imm(&mask_img, 0, 0, valid_w.max(1), valid_h.max(1)).to_image();

    let smoothed_low_res = image::imageops::blur(&low_res_cropped, 1.5);

    let guide_rgb = high_res_guide.to_rgb8();
    let dynamic_radius = (orig_width.max(orig_height) / 200).max(8) as usize;

    let mut final_mask = fast_color_guided_filter(
        &guide_rgb,
        &smoothed_low_res,
        dynamic_radius,
        1e-3,
        2.5,
        0.50,
    );

    let mut lut = [0u8; 256];
    for i in 0..=255 {
        if i <= 8 {
            lut[i] = 0;
        } else if i >= 247 {
            lut[i] = 255;
        } else {
            lut[i] = (((i as f32 - 8.0) / (247.0 - 8.0)) * 255.0).round() as u8;
        }
    }

    for p in final_mask.pixels_mut() {
        p[0] = lut[p[0] as usize];
    }

    let feathered_mask = image::imageops::blur(&final_mask, 1.0);

    Ok(feathered_mask)
}

pub fn run_sky_seg_model(
    image: &DynamicImage,
    sky_seg_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let resized_image = image.resize(SKYSEG_INPUT_SIZE, SKYSEG_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((SKYSEG_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((SKYSEG_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, SKYSEG_INPUT_SIZE as usize, SKYSEG_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = sky_seg_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in out_slice {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let usize_size = SKYSEG_INPUT_SIZE as usize;
    let mut cropped_mask_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_mask_data.push(pixel);
        }
    }

    let cropped_mask = GrayImage::from_raw(resized_w, resized_h, cropped_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Sky Segmentation output"))?;

    let smoothed_mask = image::imageops::blur(&cropped_mask, 1.5);

    let guide_rgb = image.to_rgb8();
    let (w, h) = image.dimensions();
    let dynamic_radius = (w.max(h) / 200).max(8) as usize;

    let mut final_mask =
        fast_color_guided_filter(&guide_rgb, &smoothed_mask, dynamic_radius, 1e-3, 2.5, 0.50);

    let mut lut = [0u8; 256];
    for i in 0..=255 {
        if i <= 8 {
            lut[i] = 0;
        } else if i >= 247 {
            lut[i] = 255;
        } else {
            lut[i] = (((i as f32 - 8.0) / (247.0 - 8.0)) * 255.0).round() as u8;
        }
    }

    for p in final_mask.pixels_mut() {
        p[0] = lut[p[0] as usize];
    }

    Ok(final_mask)
}

pub fn run_u2netp_model(
    image: &DynamicImage,
    u2netp_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let resized_image = image.resize(U2NETP_INPUT_SIZE, U2NETP_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((U2NETP_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((U2NETP_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, U2NETP_INPUT_SIZE as usize, U2NETP_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = u2netp_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in out_slice {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let usize_size = U2NETP_INPUT_SIZE as usize;
    let mut cropped_mask_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_mask_data.push(pixel);
        }
    }

    let cropped_mask = GrayImage::from_raw(resized_w, resized_h, cropped_mask_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from U-2-Netp output"))?;

    let smoothed_mask = image::imageops::blur(&cropped_mask, 1.5);

    let guide_rgb = image.to_rgb8();
    let (w, h) = image.dimensions();
    let dynamic_radius = (w.max(h) / 200).max(8) as usize;

    let mut final_mask =
        fast_color_guided_filter(&guide_rgb, &smoothed_mask, dynamic_radius, 1e-3, 2.5, 0.50);

    let mut lut = [0u8; 256];
    for i in 0..=255 {
        if i <= 8 {
            lut[i] = 0;
        } else if i >= 247 {
            lut[i] = 255;
        } else {
            lut[i] = (((i as f32 - 8.0) / (247.0 - 8.0)) * 255.0).round() as u8;
        }
    }

    for p in final_mask.pixels_mut() {
        p[0] = lut[p[0] as usize];
    }

    Ok(final_mask)
}

pub fn run_depth_anything_model(
    image: &DynamicImage,
    depth_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let resized_image = image.resize(DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((DEPTH_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((DEPTH_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, DEPTH_INPUT_SIZE as usize, DEPTH_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = depth_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let usize_size = DEPTH_INPUT_SIZE as usize;

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let mut cropped_depth_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_depth_data.push(pixel);
        }
    }

    let depth_map = GrayImage::from_raw(resized_w, resized_h, cropped_depth_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Depth output"))?;

    Ok(depth_map)
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiSubjectMaskParameters {
    #[serde(default)]
    pub text_prompt: String,
    #[serde(default)]
    pub use_box: bool,
    pub start_x: f64,
    pub start_y: f64,
    pub end_x: f64,
    pub end_y: f64,
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiSkyMaskParameters {
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiForegroundMaskParameters {
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiDepthMaskParameters {
    #[serde(default)]
    pub min_depth: f32,
    #[serde(default)]
    pub max_depth: f32,
    #[serde(default)]
    pub min_fade: f32,
    #[serde(default)]
    pub max_fade: f32,
    #[serde(default)]
    pub feather: f32,
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}
