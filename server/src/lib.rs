use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::thread;

use tauri::Manager as _;

mod ai;
mod config;
pub mod game;
mod listen;
mod logger;
pub mod oracle;
pub mod solver;
mod vision;
mod worker;

// 全局共享状态
struct SharedState {
    config: Arc<RwLock<config::Config>>,
    listen_thread: Mutex<Option<thread::JoinHandle<()>>>,
    calib_request: Mutex<Option<worker::CalibRequest>>,
    recalib_request: Mutex<bool>,
}

static SHARED_STATE: OnceLock<SharedState> = OnceLock::new();

// 提示档位版本号: 切换档位/我方阵营时递增, 监听线程比对发现变化后
// 即使棋盘未变也立即重算当前局面提示
static HINT_LEVEL_VERSION: AtomicU32 = AtomicU32::new(0);

pub fn bump_hint_version() {
    HINT_LEVEL_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub fn hint_version() -> u32 {
    HINT_LEVEL_VERSION.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            logger::init_tracer(tracing::Level::DEBUG, &app.path().app_config_dir().unwrap());

            let _ = SHARED_STATE.get_or_init(|| {
                let config = config::Config::load(&app.path().config_dir().unwrap());
                SharedState {
                    config: Arc::new(RwLock::new(config)),
                    listen_thread: Mutex::new(None),
                    calib_request: Mutex::new(None),
                    recalib_request: Mutex::new(false),
                }
            });

            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            listen::list_windows,
            worker::start_listen,
            worker::stop_listen,
            worker::first_frame_dataurl,
            worker::calibrate,
            worker::recalibrate,
            config::get_engine_config,
            config::set_engine_depth,
            config::set_engine_time,
            config::set_engine_alt_score_gap,
            config::set_engine_hint_level,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
