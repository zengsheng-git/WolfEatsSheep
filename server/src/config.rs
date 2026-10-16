use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use serde::Deserialize;
use serde::Serialize;
use tracing::debug;

use crate::SHARED_STATE;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct EngineConfig {
    pub depth: usize,       // 搜索深度
    pub time: usize,        // 搜索时间上限(毫秒)
    pub alt_score_gap: isize, // 次优着法分差阈值
    pub hint_level: i32,    // 提示强度 0-5(0=跟随配置)
    pub my_camp: char,      // 我方阵营 'w'=炮 'b'=兵
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { depth: 6, time: 1000, alt_score_gap: 100, hint_level: 0, my_camp: 'w' }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(skip)]
    config_path: Option<PathBuf>,
    // trace, debug, info, warn, silent
    pub loglevel: String,
    pub timer_interval: u64,  // 监听截图间隔(毫秒)
    pub confirm_interval: u64, // 棋盘变化确认间隔(毫秒)

    pub engine: EngineConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            config_path: None,
            loglevel: "INFO".to_string(),
            timer_interval: 200,
            confirm_interval: 200,
            engine: Default::default(),
        }
    }
}

impl Config {
    pub fn load(base: &Path) -> Self {
        let dir = base.join("sanpao");
        if !dir.exists() {
            let _ = fs::create_dir(&dir);
        };

        let config_path = dir.join("config.json");
        debug!("load config from '{}'", config_path.display());

        if config_path.exists() {
            let reader = BufReader::new(File::open(&config_path).unwrap());
            if let Ok(mut config) = serde_json::from_reader::<_, Config>(reader) {
                match config.config_path {
                    Some(ref path) => {
                        if path != &config_path {
                            config.config_path = Some(config_path);
                            config.save();
                        }
                    }
                    None => {
                        config.config_path = Some(config_path);
                        config.save();
                    }
                };
                return config;
            };

            // 解析失败代表配置不兼容, 删除后使用默认配置
            std::fs::remove_file(&config_path).unwrap();
            debug!("remove old config '{}'", config_path.display())
        }

        let config = Config { config_path: Some(config_path), ..Default::default() };
        config.save();
        config
    }

    pub fn save(&self) {
        let path = self.config_path.as_ref().unwrap();
        debug!("save config to '{}'", path.display());
        let json_string = serde_json::to_string_pretty(self).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(json_string.as_bytes()).unwrap();
    }
}

#[tauri::command]
pub async fn get_engine_config() -> EngineConfig {
    SHARED_STATE.get().unwrap().config.read().unwrap().engine
}

#[tauri::command]
pub async fn set_engine_depth(depth: usize) {
    let state = SHARED_STATE.get().unwrap();
    let mut config = state.config.write().unwrap();
    config.engine.depth = depth.max(1);
    config.save();
    debug!("set_engine_depth: {}", depth);
}

#[tauri::command]
pub async fn set_engine_time(time: f32) {
    let state = SHARED_STATE.get().unwrap();
    let mut config = state.config.write().unwrap();
    // 界面单位为秒, 保存为毫秒
    config.engine.time = (time * 1000.0) as usize;
    config.save();
    debug!("set_engine_time: {}", time);
}

#[tauri::command]
pub async fn set_engine_alt_score_gap(gap: isize) {
    let state = SHARED_STATE.get().unwrap();
    let mut config = state.config.write().unwrap();
    config.engine.alt_score_gap = gap;
    config.save();
    debug!("set_engine_alt_score_gap: {}", gap);
}

#[tauri::command]
pub async fn set_engine_hint_level(level: i32) {
    let state = SHARED_STATE.get().unwrap();
    let mut config = state.config.write().unwrap();
    config.engine.hint_level = level.clamp(0, 5);
    config.save();
    debug!("set_engine_hint_level: {}", level);
    // 递增版本号通知监听线程: 即使棋盘未变也按新档位重算当前局面
    crate::bump_hint_version();
}

#[tauri::command]
pub async fn set_my_camp(camp: String) {
    let state = SHARED_STATE.get().unwrap();
    let mut config = state.config.write().unwrap();
    if camp == "w" || camp == "b" {
        config.engine.my_camp = camp.chars().next().unwrap();
        config.save();
        debug!("set_my_camp: {}", camp);
        // 箭头着色随我方阵营变化, 重算提示立即生效
        crate::bump_hint_version();
    }
}
