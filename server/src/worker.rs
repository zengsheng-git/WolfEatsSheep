// 监听分析循环: 截图 -> 识别局面 -> 走子检测 -> AI 分析 -> 事件推送
// 状态机沿用原项目(象棋助手)的结构, 识别与引擎替换为三炮十五兵实现
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::AppHandle;
use tauri::Emitter as _;
use tauri::Manager as _;
use tracing::{debug, error, info, trace, warn};
use xcap::image::{ImageBuffer, Rgba};

use crate::ai::{self, SearchLimits};
use crate::config::EngineConfig;
use crate::game::{self, Board, Camp, Changed, Position};
use crate::listen::{ListenWindow, Window};
use crate::vision::{self, GridModel};
use crate::SHARED_STATE;

#[derive(Serialize, Clone, Debug)]
pub struct DeviationCost {
    pub camp: char,
    pub loss: isize,
}

// 分析事件负载, 字段与前端 Analyse 面板一一对应
#[derive(Serialize, Clone, Debug)]
pub struct AnalyseResult {
    pub depth: usize,
    pub score: isize,           // 行棋方视角
    pub time: usize,            // 毫秒
    pub pvs: Vec<String>,       // 最佳主线(iccs)
    pub moves: Vec<String>,     // 主线中文记谱
    pub alternatives: Vec<String>, // 次优首着(iccs)
    pub alt_scores: Vec<isize>, // 次优与最优分差
    pub alt_moves: Vec<String>, // 次优中文记谱
    pub winrate: Option<usize>, // 预留, 当前为空
    pub deviation: Option<DeviationCost>,
    pub board: Vec<Position>,   // 分析时局面, 供主线预演
    pub state: String,
    pub source: String,
    pub camp: char,             // 行棋方 'w'炮 'b'兵
    pub over: Option<String>,   // 终局文案
}

struct BoardAnalysisResult {
    expect_move: Changed,
    expect_board: Board,
}

// 手动纠错请求: 用户在镜像棋盘上标注的局面与标注后的行棋方
pub struct CalibRequest {
    pub board: Board,
    pub turn: Camp,
}

#[derive(PartialEq)]
enum ChessboardState {
    Initial,      // 初始状态
    StartPos,     // 开局等待首着
    OurTurn,      // 我方行棋
    OpponentTurn, // 对方行棋
    Invalid,      // 无效状态
    GameOver,     // 终局, 等待新开局
}

struct AnalysisContext {
    app: AppHandle,
    window: ListenWindow,
    model: GridModel,
    last_board: Board,
    expect_move: Changed,
    expect_board: Board,
    expect_score: isize,       // 上次分析的评分(行棋方视角)
    expect_camp: Camp,         // 上次分析的行棋方
    pending_deviation: Option<Camp>,
    invalid_change_count: usize,
    last_hint_version: u32,
    no_capture_plies: usize,   // 连续无吃子步数(和棋判定)
}

// xcap::Window 内含原生句柄, 仅在监听线程内使用, 显式声明跨线程移动安全
unsafe impl Send for AnalysisContext {}
unsafe impl Sync for AnalysisContext {}

impl AnalysisContext {
    fn new(app: AppHandle, window: ListenWindow, model: GridModel) -> Self {
        Self {
            app,
            window,
            model,
            last_board: [[game::EMPTY; game::SIZE]; game::SIZE],
            expect_move: Changed::default(),
            expect_board: [[game::EMPTY; game::SIZE]; game::SIZE],
            expect_score: 0,
            expect_camp: Camp::None,
            pending_deviation: None,
            invalid_change_count: 0,
            last_hint_version: crate::hint_version(),
            no_capture_plies: 0,
        }
    }

    fn should_stop(&self) -> bool {
        let state = SHARED_STATE.get().unwrap();
        state.listen_thread.lock().unwrap().is_none()
    }

    fn my_camp(&self) -> Camp {
        Camp::from_char(SHARED_STATE.get().unwrap().config.read().unwrap().engine.my_camp)
    }

    fn mirror(&self) -> bool {
        // 把我方翻到屏幕下方显示
        (self.my_camp() == Camp::Cannon) == self.model.cannons_top
    }

    // 确认棋盘状态是否稳定(二次截图比对)
    fn confirm_board(&self, board: Board) -> bool {
        thread::sleep(Duration::from_millis(100));
        let Some(conf_image) = self.window.capture() else { return false };
        self.get_board(conf_image) == Some(board)
    }

    fn get_board(&self, image: ImageBuffer<Rgba<u8>, Vec<u8>>) -> Option<Board> {
        vision::detect_frame(&self.model, &image)
    }

    // 分析棋盘并返回预期(建议着法走完后的局面)
    fn analyze_board(&mut self, camp: &Camp, board: Board) -> Option<BoardAnalysisResult> {
        let engine_config = {
            let config = SHARED_STATE.get().unwrap().config.read().unwrap();
            config.engine
        };
        let result = search_with_config(board, *camp, &engine_config)?;
        self.last_hint_version = crate::hint_version();

        // 上一步为非预期走子时, 携带当时的预期评分用于计算偏离代价
        let prev = self.pending_deviation.take().and_then(|dev_camp| {
            if self.expect_camp == dev_camp { Some((dev_camp, self.expect_score)) } else { None }
        });

        let score = result.score;
        let (expect_move, expect_board) = analyse(&self.app, result, board, *camp, prev, self.no_capture_plies)?;
        self.expect_score = score;
        self.expect_camp = *camp;
        Some(BoardAnalysisResult { expect_move, expect_board })
    }

    fn update_ui(&self, board: Board) {
        let board_map = game::board_map(&board);
        self.app.emit("mirror", self.mirror()).unwrap();
        self.app.emit("position", &board_map).unwrap();
    }

    fn handle_move(&mut self, changed: &Changed) {
        self.app.emit("move", changed).unwrap();
    }

    fn emit_terminal(&self, board: Board, text: &str) {
        let result = AnalyseResult {
            depth: 0,
            score: 0,
            time: 0,
            pvs: Vec::new(),
            moves: Vec::new(),
            alternatives: Vec::new(),
            alt_scores: Vec::new(),
            alt_moves: Vec::new(),
            winrate: None,
            deviation: None,
            board: game::board_map(&board),
            state: "终局".to_string(),
            source: "引擎".to_string(),
            camp: 'n',
            over: Some(text.to_string()),
        };
        info!("终局: {text}");
        self.app.emit("analyse", result).unwrap();
    }

    // 按当前档位重算给定行棋方的提示(切档后棋盘未变时立即刷新)
    fn reanalyze(&mut self, camp: &Camp) {
        if let Some(result) = self.analyze_board(camp, self.last_board) {
            self.expect_move = result.expect_move;
            self.expect_board = result.expect_board;
        }
    }

    fn analyze_and_set_state(&mut self, turn: Camp, board: Board, camp: &Camp) -> ChessboardState {
        if let Some(result) = self.analyze_board(&turn, board) {
            self.expect_move = result.expect_move;
            self.expect_board = result.expect_board;
        }
        if turn.eq(camp) { ChessboardState::OurTurn } else { ChessboardState::OpponentTurn }
    }

    // 处理一次走棋: 更新UI/无吃子计数/终局检测, 分析下一行动方并返回新状态
    fn handle_move_and_next(&mut self, changed: &Changed, board: Board, camp: &Camp) -> ChessboardState {
        self.last_board = board;
        self.handle_move(changed);
        // 吃子判定: 平移着法起落点距离恒为 1, 隔山打的落点至少隔 2 格
        let dist = chebyshev_dist(&changed.from, &changed.to);
        if dist > 1 {
            self.no_capture_plies = 0;
        } else {
            self.no_capture_plies += 1;
        }
        self.pending_deviation = Some(changed.camp);
        // 终局判定
        if let Some(winner) = game::winner(&board) {
            self.emit_terminal(board, terminal_text(winner));
            return ChessboardState::GameOver;
        }
        if self.no_capture_plies >= game::DRAW_NO_CAPTURE_PLIES {
            self.emit_terminal(board, "和棋(长期无吃子)");
            return ChessboardState::GameOver;
        }
        self.analyze_and_set_state(changed.camp.opposite(), board, camp)
    }

    fn handle_invalid_change(&mut self, last_board: Board, board: Board, camp: &Camp) -> ChessboardState {
        if self.invalid_change_count < 3 {
            self.invalid_change_count += 1;
            debug!("OneChanged last {:?} current {:?}", last_board, board);
            let _ = camp;
            ChessboardState::Invalid
        } else {
            debug!("OneChanged count=3, reload");
            self.invalid_change_count = 0;
            ChessboardState::Initial
        }
    }

    fn reset_for_new_game(&mut self, board: Board) {
        self.last_board = board;
        self.no_capture_plies = 0;
        self.invalid_change_count = 0;
        self.expect_move = Changed::default();
        self.pending_deviation = None;
    }

    // 重新识别: 用最近一次通过规则校验的局面(last_board)重采两方颜色质心,
    // 不重置对局状态 -- 用于棋盘显示正确、但识别因高亮/特效走样的场景
    fn recalibrate(&mut self) {
        let Some(image) = self.window.capture() else {
            warn!("重新识别: 截图失败");
            return;
        };
        match vision::calibrate_colors(&self.model, &image, &self.last_board) {
            Some((cc, sc)) => {
                info!(
                    "重新识别: 颜色质心 炮=({:.0},{:.0},{:.0}) 兵=({:.0},{:.0},{:.0})",
                    cc[0], cc[1], cc[2], sc[0], sc[1], sc[2]
                );
                self.model.cannon_color = cc;
                self.model.soldier_color = sc;
            }
            None => warn!("重新识别: 未采到双方颜色样本, 保持原样"),
        }
    }

    // 手动纠错: 以用户标注的局面为准重同步
    // 先用标注格上的色块重采样两方颜色质心(纠正分类漂移), 再按指定行棋方重新分析
    fn apply_calibration(&mut self, board: Board, turn: Camp) -> ChessboardState {
        if let Some(image) = self.window.capture()
            && let Some((cc, sc)) = vision::calibrate_colors(&self.model, &image, &board)
        {
            info!(
                "手动纠错: 颜色质心重校准 炮=({:.0},{:.0},{:.0}) 兵=({:.0},{:.0},{:.0})",
                cc[0], cc[1], cc[2], sc[0], sc[1], sc[2]
            );
            self.model.cannon_color = cc;
            self.model.soldier_color = sc;
        }
        self.reset_for_new_game(board);
        self.update_ui(board);
        if let Some(result) = self.analyze_board(&turn, board) {
            self.expect_move = result.expect_move;
            self.expect_board = result.expect_board;
        }
        self.last_board = board;
        let camp = self.my_camp();
        if turn.eq(&camp) {
            ChessboardState::OurTurn
        } else {
            ChessboardState::OpponentTurn
        }
    }
}

fn chebyshev_dist(from: &str, to: &str) -> usize {
    match (game::xy_of(from), game::xy_of(to)) {
        (Some((fx, fy)), Some((tx, ty))) => fx.abs_diff(tx).max(fy.abs_diff(ty)),
        _ => 0,
    }
}

fn terminal_text(winner: Camp) -> &'static str {
    match winner {
        Camp::Cannon => "炮方胜(兵已不足)",
        Camp::Soldier => "兵方胜(围困三炮)",
        Camp::None => "终局",
    }
}

// 识别长时间失败时保存当前帧, 便于按真实画面定位识别问题(同首帧诊断)
fn save_stuck_frame(app: &AppHandle, image: &ImageBuffer<Rgba<u8>, Vec<u8>>) {
    let Some(dir) = app.path().app_data_dir().ok().map(|d| d.join("sanpao")) else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("stuck_frame.png");
    if xcap::image::save_buffer(&path, image.as_raw(), image.width(), image.height(), xcap::image::ExtendedColorType::Rgba8).is_ok() {
        warn!("识别连续失败, 已保存卡住帧: {}", path.display());
    }
}

fn search_with_config(board: Board, camp: Camp, config: &EngineConfig) -> Option<ai::Analysis> {
    let depth = ai::hint_depth(config.hint_level, config.depth);
    let limits = SearchLimits { depth, time_ms: config.time as u64 };
    ai::analyze(&board, camp, limits, config.alt_score_gap)
}

// 将搜索结果整理为完整展示数据(中文主线/次优/局面/偏离/终局), 并推送给前端
fn analyse(
    app: &AppHandle,
    result: ai::Analysis,
    board: Board,
    camp: Camp,
    prev: Option<(Camp, isize)>,
    no_capture_plies: usize,
) -> Option<(Changed, Board)> {
    let Some(best_pv) = result.pvs.first().cloned() else {
        warn!("分析结果缺少着法, 跳过展示");
        return None;
    };

    let mut payload = AnalyseResult {
        depth: result.depth,
        score: result.score,
        time: result.time_ms as usize,
        pvs: result.pvs.clone(),
        moves: Vec::new(),
        alternatives: result.alternatives.clone(),
        alt_scores: result.alt_scores.clone(),
        alt_moves: Vec::new(),
        winrate: None,
        deviation: None,
        board: game::board_map(&board),
        state: "分析".to_string(),
        source: "引擎".to_string(),
        camp: camp.to_char(),
        over: None,
    };

    // 上一步为非预期走子时, 计算相对预期的亏损(前后评分均为行棋方视角)
    if let Some((dev_camp, prev_score)) = prev
        && dev_camp == camp.opposite()
    {
        payload.deviation = Some(DeviationCost { camp: dev_camp.to_char(), loss: prev_score + result.score });
    }

    // 终局预警: 距离吃光/围死不远时提示
    let over = if let Some(winner) = game::winner(&board) {
        Some(terminal_text(winner).to_string())
    } else if no_capture_plies + 10 >= game::DRAW_NO_CAPTURE_PLIES {
        Some("接近和棋(长期无吃子)".to_string())
    } else {
        None
    };
    payload.over = over;

    // 主线中文记谱(逐着重放)
    payload.moves.push(game::move_text(&board, &best_pv));
    let mut tmp_board = game::apply_move(&board, &best_pv);
    for pv in result.pvs.iter().skip(1) {
        payload.moves.push(game::move_text(&tmp_board, pv));
        tmp_board = game::apply_move(&tmp_board, pv);
    }
    for alt in result.alternatives.iter() {
        payload.alt_moves.push(game::move_text(&board, alt));
    }

    let expect_board = game::apply_move(&board, &best_pv);
    let (fx, fy) = game::xy_of(&best_pv[..2])?;
    let expect_move = Changed {
        piece: board[fy][fx],
        from: best_pv[..2].to_string(),
        to: best_pv[2..].to_string(),
        camp,
    };

    info!("分析结果 score={} pvs={:?}", payload.score, payload.pvs);
    app.emit("analyse", payload).unwrap();
    Some((expect_move, expect_board))
}

// 处理循环: 包一层捕获 panic, 线程内异常只终止本次监听, 不连锁影响界面
fn process_analysis_loop_guarded(context: AnalysisContext) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_analysis_loop(context);
    }));
    if let Err(e) = result {
        let msg = e.downcast_ref::<String>().map(|s| s.as_str())
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("未知错误");
        error!("分析线程异常退出: {msg}");
    }
}

fn process_analysis_loop(mut context: AnalysisContext) {
    let mut current_state = ChessboardState::Initial;
    let mut capture_failures = 0usize;
    let mut bad_frames = 0usize;
    let mut last_invalid_dump: Option<Instant> = None;

    loop {
        if context.should_stop() {
            debug!("listen stopped");
            break;
        }

        let interval = SHARED_STATE.get().unwrap().config.read().unwrap().timer_interval;
        thread::sleep(Duration::from_millis(interval));

        // 手动纠错请求: 以标注局面与颜色样本重同步
        if let Some(req) = SHARED_STATE.get().unwrap().calib_request.lock().unwrap().take() {
            current_state = context.apply_calibration(req.board, req.turn);
            continue;
        }

        // 重新识别请求: 按当前已知局面重学两方颜色特征
        if {
            let mut flag = SHARED_STATE.get().unwrap().recalib_request.lock().unwrap();
            std::mem::take(&mut *flag)
        } {
            context.recalibrate();
            continue;
        }

        let Some(image) = context.window.capture() else {
            capture_failures += 1;
            if capture_failures >= 3 {
                error!("目标窗口连续截图失败, 可能已关闭, 停止监听");
                let _ = context.app.emit("listen_stopped", "目标窗口可能已关闭, 监听已停止");
                break;
            }
            continue;
        };
        capture_failures = 0;

        let board = context.get_board(image.clone()).filter(|b| game::board_check(b));
        let Some(board) = board else {
            // 识别失败/局面不合理: 短暂抖动属正常, 持续失败则保存现场供诊断
            bad_frames += 1;
            if bad_frames == 20 || bad_frames % 200 == 0 {
                save_stuck_frame(&context.app, &image);
            }
            continue;
        };
        bad_frames = 0;
        let camp = context.my_camp();

        current_state = match current_state {
            ChessboardState::Initial => {
                debug!("首次启动，立即分析");
                context.update_ui(board);
                // 开局炮先行; 中途接入时按我方回合近似
                let turn = if game::is_startpos(&board) { Camp::Cannon } else { camp };
                if let Some(result) = context.analyze_board(&turn, board) {
                    context.expect_move = result.expect_move;
                    context.expect_board = result.expect_board;
                }
                context.last_board = board;
                if game::is_startpos(&board) {
                    ChessboardState::StartPos
                } else if turn.eq(&camp) {
                    ChessboardState::OurTurn
                } else {
                    ChessboardState::OpponentTurn
                }
            }

            ChessboardState::StartPos => {
                if !game::is_startpos(&board) {
                    if board == context.last_board {
                        ChessboardState::StartPos
                    } else {
                        handle_board_change(&mut context, board, &camp)
                    }
                } else if board != context.last_board {
                    // 同为开局但镜像翻转(对方执炮): 重置基准
                    context.update_ui(board);
                    context.last_board = board;
                    ChessboardState::StartPos
                } else {
                    // 仍是开局等待首着: 切档后立即按新档位重算
                    reanalyze_if_hint_changed(&mut context, &camp);
                    ChessboardState::StartPos
                }
            }

            ChessboardState::OurTurn | ChessboardState::OpponentTurn => {
                if game::is_startpos(&board) && !game::is_startpos(&context.last_board) {
                    // 新开局(上局结束后重开)
                    debug!("检测到新开局, 重置上下文");
                    context.reset_for_new_game(board);
                    context.update_ui(board);
                    ChessboardState::Initial
                } else if board == context.last_board {
                    reanalyze_if_hint_changed(&mut context, &camp);
                    current_state
                } else if board == context.expect_board {
                    // 走了建议着法: 直接复用, 分析下一行动方
                    debug!("棋盘为预期棋盘，分析下一行动方");
                    let expect_move = context.expect_move.clone();
                    let expect_board = context.expect_board;
                    let mover = expect_move.camp;
                    context.last_board = expect_board;
                    context.handle_move(&expect_move);
                    context.analyze_and_set_state(mover.opposite(), expect_board, &camp)
                } else {
                    if !context.confirm_board(board) {
                        debug!("棋盘延迟确认失败");
                        let confirm_interval = SHARED_STATE.get().unwrap().config.read().unwrap().confirm_interval;
                        thread::sleep(Duration::from_millis(confirm_interval));
                        current_state
                    } else {
                        handle_board_change(&mut context, board, &camp)
                    }
                }
            }

            ChessboardState::Invalid => ChessboardState::Initial,

            ChessboardState::GameOver => {
                if game::is_startpos(&board) && !game::is_startpos(&context.last_board) {
                    debug!("终局后检测到新开局");
                    context.reset_for_new_game(board);
                    context.update_ui(board);
                    ChessboardState::Initial
                } else {
                    ChessboardState::GameOver
                }
            }
        };

        // 非法走法(疑似误读)时保存现场供诊断(3 秒节流, 避免连续误读刷屏)
        if current_state == ChessboardState::Invalid
            && last_invalid_dump.map_or(true, |t| t.elapsed() >= Duration::from_secs(3))
        {
            last_invalid_dump = Some(Instant::now());
            save_stuck_frame(&context.app, &image);
        }
    }
}

// 切档后棋盘未变时立即按新档位重算
fn reanalyze_if_hint_changed(context: &mut AnalysisContext, camp: &Camp) {
    let version = crate::hint_version();
    if version != context.last_hint_version {
        context.last_hint_version = version;
        info!("提示档位已切换, 按新档位重算当前局面");
        context.reanalyze(camp);
    } else {
        debug!("棋盘未发生变化，跳过分析");
    }
}

// 处理一步实际变化: 差分 + 合法性校验, 分发到对应状态
fn handle_board_change(context: &mut AnalysisContext, board: Board, camp: &Camp) -> ChessboardState {
    let (changed, board_state) = game::board_diff(&context.last_board, &board);
    match board_state {
        game::BoardChangeState::Move => {
            let mv = format!("{}{}", changed.from, changed.to);
            if !game::is_legal(&context.last_board, &mv) {
                debug!("检测到非法着法(疑似误读): {mv}");
                context.handle_invalid_change(context.last_board, board, camp)
            } else {
                context.handle_move_and_next(&changed, board, camp)
            }
        }
        game::BoardChangeState::One => context.handle_invalid_change(context.last_board, board, camp),
        game::BoardChangeState::Unknown => {
            debug!("棋局变化未知，重置上下文");
            context.update_ui(board);
            context.last_board = board;
            ChessboardState::Initial
        }
    }
}

#[tauri::command]
pub async fn start_listen(app: AppHandle, target: Window) -> Result<(), String> {
    trace!("start_listen");
    {
        let shared_state = SHARED_STATE.get().unwrap();
        let mut listen_slot = shared_state.listen_thread.lock().unwrap();
        if listen_slot.is_some() {
            error!("current listen thread is running, please stop it first");
            return Err("已经在监听中".to_string());
        }

        let Some(mut window) = ListenWindow::new(&target, 0, 0) else {
            return Err("目标窗口不存在".to_string());
        };
        let Some(image) = window.capture() else {
            return Err("无法截取目标窗口画面".to_string());
        };

        // 首帧必须识别出标准开局, 由此确定网格几何与两方颜色
        let detect = match vision::detect_first(&image) {
            Ok(det) => det,
            Err(e) => {
                // 识别失败时保存首帧截图, 便于按真实画面调优识别参数
                let shot_path = app
                    .path()
                    .app_data_dir()
                    .map(|d| d.join("sanpao").join("first_frame.png"))
                    .unwrap_or_else(|_| std::env::temp_dir().join("sanpao_first_frame.png"));
                let _ = std::fs::create_dir_all(shot_path.parent().unwrap());
                let saved = xcap::image::save_buffer(
                    &shot_path,
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    xcap::image::ExtendedColorType::Rgba8,
                )
                .is_ok();
                let hint = if saved {
                    format!("(首帧截图已保存: {})", shot_path.display())
                } else {
                    String::new()
                };
                return Err(format!("{e} {hint}"));
            }
        };
        trace!("首帧局面: {:?}", detect.board);
        window.set_sub_bound(detect.crop.0, detect.crop.1, detect.crop.2, detect.crop.3);
        info!("识别到开局棋盘, crop={:?} cannons_top={}", detect.crop, detect.model.cannons_top);

        let context = AnalysisContext::new(app.clone(), window, detect.model);
        let listen_thread = thread::spawn(move || {
            trace!("into thread");
            process_analysis_loop_guarded(context);
        });
        *listen_slot = Some(listen_thread);
    }
    Ok(())
}

#[tauri::command]
pub fn stop_listen() {
    info!("stop listen");
    let shared_state = SHARED_STATE.get().unwrap();
    if let Ok(mut state) = shared_state.listen_thread.lock()
        && let Some(listen_thread) = state.take()
    {
        drop(state);
        // join 移到独立线程, 停止命令立即返回
        thread::spawn(move || {
            if let Err(e) = listen_thread.join() {
                error!("等待监听线程退出失败: {:?}", e);
            }
        });
    }
    debug!("stoped");
}

// 手动纠错: board 为 5 行字符串(每行 5 字符, 'C'炮/'S'兵/' '空), turn 'w'炮/'b'兵
#[tauri::command]
pub fn calibrate(board: Vec<String>, turn: String) -> Result<(), String> {
    let Some(parsed) = parse_board(&board) else {
        return Err("标注数据格式错误".to_string());
    };
    if !game::board_check(&parsed) {
        let (cannons, soldiers) = game::counts(&parsed);
        return Err(format!("标注局面不合法: 需要 3 门炮(当前 {cannons}), 兵不超过 15(当前 {soldiers})"));
    }
    let turn_camp = match turn.as_str() {
        "w" => Camp::Cannon,
        "b" => Camp::Soldier,
        _ => return Err("行棋方参数错误".to_string()),
    };
    let shared_state = SHARED_STATE.get().unwrap();
    if shared_state.listen_thread.lock().unwrap().is_none() {
        return Err("尚未开始监听".to_string());
    }
    *shared_state.calib_request.lock().unwrap() = Some(CalibRequest { board: parsed, turn: turn_camp });
    info!("收到手动纠错请求");
    Ok(())
}

fn parse_board(rows: &[String]) -> Option<Board> {
    if rows.len() != game::SIZE {
        return None;
    }
    let mut board = [[game::EMPTY; game::SIZE]; game::SIZE];
    for (y, row) in rows.iter().enumerate() {
        let chars: Vec<char> = row.chars().collect();
        if chars.len() != game::SIZE {
            return None;
        }
        for (x, c) in chars.iter().enumerate() {
            board[y][x] = match c {
                'C' => game::CANNON,
                'S' => game::SOLDIER,
                ' ' => game::EMPTY,
                _ => return None,
            };
        }
    }
    Some(board)
}

// 重新识别: 按当前已知局面重学炮/兵颜色特征(修复高亮/特效导致的识别走样)
#[tauri::command]
pub fn recalibrate() -> Result<(), String> {
    let shared_state = SHARED_STATE.get().unwrap();
    if shared_state.listen_thread.lock().unwrap().is_none() {
        return Err("尚未开始监听".to_string());
    }
    *shared_state.recalib_request.lock().unwrap() = true;
    info!("收到重新识别请求");
    Ok(())
}
