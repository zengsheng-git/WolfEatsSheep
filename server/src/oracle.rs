// 完美着法查询: 读取 solver 生成的分层的值表, 任意局面直接给出严格最优着法
// (炮胜局面选致胜着, 兵胜局面选致胜着, 和棋局面选保和着, 必败局面选同类着法作最长抵抗)
// 表缺失时返回 None, 调用方回落到搜索引擎
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::ai::{self, Analysis};
use crate::game::{self, Board, Camp};
use crate::solver::{self, VAL_SHEEP, VAL_WOLF};

fn tables_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SANPAO_TABLES") {
        return Some(PathBuf::from(dir));
    }
    // 便携模式: exe 同目录下的 tables/ (打包分发时把表随程序一起拷贝即可)
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        let dir = parent.join("tables");
        if dir.is_dir() {
            return Some(dir);
        }
    }
    // 与 config.rs 的持久化目录一致: Windows 上 tauri config_dir = %APPDATA%
    std::env::var("APPDATA").ok().map(|p| PathBuf::from(p).join("sanpao").join("tables"))
}

// 表用内存映射(mmap)加载: 不占堆内存, 由操作系统按需换页并在内存紧张时自动回收,
// 顺便让"缓存两层大表"不再带来 GB 级常驻内存
fn layer_table(k: usize) -> Option<Arc<memmap2::Mmap>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<usize, Arc<memmap2::Mmap>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(t) = guard.get(&k) {
        return Some(t.clone());
    }
    let dir = tables_dir()?;
    let (per_side, _) = solver::layer_counts(k);
    let expect = ((per_side * 2 + 3) / 4) as u64;
    let file = std::fs::File::open(dir.join(solver::layer_filename(k))).ok()?;
    let meta = file.metadata().ok()?;
    if meta.len() != expect {
        return None;
    }
    // safety: 表文件为一次性生成的只读数据, 运行期间不会被写入
    let table = Arc::new(unsafe { memmap2::Mmap::map(&file).ok()? });
    guard.insert(k, table.clone());
    // 只保留最近两层(一层行棋参考 + 一层吃子降层过渡)
    while guard.len() > 2 {
        let oldest = guard.keys().copied().min()?;
        if oldest == k {
            break;
        }
        guard.remove(&oldest);
    }
    Some(table)
}

fn to_solver_board(board: &Board) -> [u8; 25] {
    let mut out = [0u8; 25];
    for y in 0..game::SIZE {
        for x in 0..game::SIZE {
            out[y * 5 + x] = match board[y][x] {
                game::CANNON => 1,
                game::SOLDIER => 2,
                _ => 0,
            };
        }
    }
    out
}

// 行棋方视角归类: 1=行棋方胜 0=和 -1=行棋方负
fn cls_of(val: u8, camp: Camp) -> i32 {
    match val {
        VAL_WOLF => if camp == Camp::Cannon { 1 } else { -1 },
        VAL_SHEEP => if camp == Camp::Soldier { 1 } else { -1 },
        _ => 0,
    }
}

// 局面的表值。None = 终局/兵数超范围/无表
pub fn state_value(board: &Board, camp: Camp) -> Option<u8> {
    if game::winner(board).is_some() {
        return None;
    }
    let (_, soldiers) = game::counts(board);
    if !(4..=15).contains(&soldiers) {
        return None;
    }
    let table = layer_table(soldiers)?;
    let side = if camp == Camp::Cannon { 0 } else { 1 };
    Some(solver::table_value(&table[..], soldiers, &to_solver_board(board), side))
}

// 该着法导致的子局面表值。子局面的兵数可能与根局面不同(吃子 -1),
// 必须按子局面自己的兵数加载对应层表, 否则索引公式与表长不匹配会越界
fn move_value(board: &Board, camp: Camp, mv: &str) -> Option<u8> {
    let child = game::apply_move(board, mv);
    let k2 = game::counts(&child).1;
    if k2 == 3 {
        return Some(solver::VAL_WOLF); // 吃到剩 3 兵即终局炮胜(表无 k=3 以下)
    }
    if !(4..=15).contains(&k2) {
        return None;
    }
    let table = layer_table(k2)?;
    let child_side = if camp == Camp::Cannon { 1 } else { 0 };
    Some(solver::table_value(&table[..], k2, &to_solver_board(&child), child_side))
}

fn is_cap(board: &Board, mv: &str) -> bool {
    game::xy_of(&mv[2..]).map(|(x, y)| board[y][x] != game::EMPTY).unwrap_or(false)
}

// 同值类着法的实用排序。值类(胜/和/负)严格优先, 同类内按阵营与值类挑"最合理"的一步:
//   - 炮方: 吃子优先(推进终局 + 重置无吃子计数), 之后按静态评估(兵越少越好/活动度);
//   - 兵方胜势(cls=+1): 进攻评估优先(压缩炮活动度、逼近围堵), 推动向终局收敛;
//   - 兵方非胜势(和/败): 纯防守——对方可吃着法数、受威胁兵数越少越好。
//     和棋局面里保和着法往往全部自带被吃敞口(必须挡在炮的跳吃线上才守得住),
//     此时"少送一个是一个"比任何进攻姿态都实际。
// 排序只在同类内挑选, 不改变任何局面的理论结果
fn ranked_moves(board: &Board, camp: Camp) -> Vec<(String, i32)> {
    let mut list: Vec<(String, i32, bool, usize, usize, isize)> = game::legal_moves(board, &camp)
        .into_iter()
        .map(|mv| {
            // 子层表缺失时按和棋类兜底(完整表下不会发生)
            let cls = move_value(board, camp, &mv).map(|v| cls_of(v, camp)).unwrap_or(0);
            let child = game::apply_move(board, &mv);
            // 对方走后立即可吃的着法数(敞口) 与 受威胁的不同兵数(每回合至多被吃一枚)
            let opp_moves = game::legal_moves(&child, &camp.opposite());
            let opp_caps = opp_moves.iter().filter(|m| is_cap(&child, m)).count();
            let mut threatened: Vec<(usize, usize)> = Vec::new();
            if camp == Camp::Soldier {
                for m in &opp_moves {
                    if is_cap(&child, m)
                        && let Some((tx, ty)) = game::xy_of(&m[2..])
                        && !threatened.contains(&(tx, ty))
                    {
                        threatened.push((tx, ty));
                    }
                }
            }
            let ev = -ai::static_eval(&child, camp.opposite());
            let cap = is_cap(board, &mv);
            (mv, cls, cap, opp_caps, threatened.len(), ev)
        })
        .collect();
    list.sort_by(|a, b| {
        let key = |e: &(String, i32, bool, usize, usize, isize)| {
            // 键: 全序降序比较。实测结论(见 experiment_sheep_hint_quality):
            //  - 兵方和棋/败势: 只看防守(敞口、受威胁兵数升序), 掺入进攻评估会主动
            //    站上跳吃线导致持续送子;  - 兵方胜势: 只看进攻评估, 安全优先会让主线
            //    无法收敛到终局(兵胜需主动弃子围堵);  - 炮方: 吃子优先后按评估
            match (camp, e.1) {
                (Camp::Soldier, cls) if cls <= 0 => (cls, 0isize, -(e.3 as isize), -(e.4 as isize), 0),
                (Camp::Soldier, _) => (e.1, 1isize, e.5, 0, 0),
                (Camp::Cannon, _) => (e.1, 1isize + e.2 as isize, e.5, 0, 0),
                (Camp::None, _) => (e.1, 0, 0, 0, 0),
            }
        };
        key(b).cmp(&key(a))
    });
    list.into_iter().map(|(mv, cls, _, _, _, _)| (mv, cls)).collect()
}

// 完美分析: 返回与搜索引擎同构的结果, 分数映射为杀棋分便于前端展示
pub fn analyze(board: &Board, camp: Camp) -> Option<Analysis> {
    if game::winner(board).is_some() {
        return None;
    }
    let (_, soldiers) = game::counts(board);
    if !(4..=15).contains(&soldiers) {
        return None;
    }
    let val = cls_of(state_value(board, camp)?, camp);

    let scored = ranked_moves(board, camp);
    let Some((best_mv, best_cls)) = scored.first().cloned() else {
        return None;
    };

    // 主线: 双方沿同类最优着法推演到终局, 遇重复局面即停。
    // 同类内优先走向"未出现过的子局面"(防自循环), 否则和棋/慢胜主线几步就重复截断
    let mut pv = Vec::new();
    let mut cur = *board;
    let mut cur_camp = camp;
    let mut seen: HashSet<([u8; 25], char)> = HashSet::new();
    for _ in 0..400 {
        if game::winner(&cur).is_some() {
            break;
        }
        if !seen.insert((to_solver_board(&cur), cur_camp.to_char())) {
            break;
        }
        let ranked = ranked_moves(&cur, cur_camp);
        let Some((top_mv, top_cls)) = ranked.first().cloned() else { break };
        let fresh = ranked
            .iter()
            .take_while(|(_, c)| *c == top_cls)
            .find(|(m, _)| {
                let child = game::apply_move(&cur, m);
                !seen.contains(&(to_solver_board(&child), cur_camp.opposite().to_char()))
            });
        let mv = fresh.map(|(m, _)| m.clone()).unwrap_or(top_mv);
        pv.push(mv.clone());
        cur = game::apply_move(&cur, &mv);
        cur_camp = cur_camp.opposite();
    }

    // 次优: 与最优同值类的着法(按同类排序给出, 靠前的更"稳")
    let mut alternatives = Vec::new();
    let mut alt_scores = Vec::new();
    for (mv, cls) in scored.iter() {
        if *cls == best_cls && *mv != best_mv && alternatives.len() < 4 {
            alternatives.push(mv.clone());
            alt_scores.push(0);
        }
    }

    // 分数映射: 主线推到终局时给出确切步数(胜 = MATE - 主线步数, 负对称, 和 = 0);
    // 值类为胜/负但主线因重复局面截断时, 步数未知, 用 ±MATE 哨兵标记
    // (引擎路径的杀棋分永远是 MATE - n, n ≥ 1, 不会与哨兵冲突)
    let terminal = game::winner(&cur).is_some();
    let score = match val {
        1 if terminal => ai::MATE - pv.len() as isize,
        1 => ai::MATE,
        -1 if terminal => -(ai::MATE - pv.len() as isize),
        -1 => -ai::MATE,
        _ => 0,
    };
    Some(Analysis {
        depth: 99,
        score,
        time_ms: 0,
        pvs: if pv.is_empty() { vec![best_mv.clone()] } else { pv },
        alternatives,
        alt_scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 回归: 开局(15兵)枚举吃子着法时子局面为 14 兵, 曾误用根层表索引导致越界 panic。
    // 表齐备时应返回破解唯一正招 c0c2; 无表环境静默跳过。
    #[test]
    fn test_oracle_startpos_capture_children() {
        let board = game::middle_startpos_board();
        let Some(a) = analyze(&board, Camp::Cannon) else {
            return; // 表缺失, 跳过
        };
        assert_eq!(a.pvs[0], "c0c2", "中间变体开局唯一不亏着法应为中炮跳吃");
        assert_eq!(a.score, 0, "开局为和棋, 分数应为 0, 实际: {}", a.score);
        let mut cur = board;
        let mut camp = Camp::Cannon;
        for mv in a.pvs.iter() {
            assert!(game::is_legal(&cur, mv), "主线着法 {mv} 非法");
            cur = game::apply_move(&cur, mv);
            camp = camp.opposite();
        }
    }

    // 临时诊断: 兵方提示质量 — 统计推荐着法的"送子敞口"(走后炮方立即可吃)与可替代性
    #[test]
    #[ignore]
    fn experiment_sheep_hint_quality() {
        // 场景 A: 中间变体主线 1.c0c2 (理论 和棋)
        let a = game::apply_move(&game::middle_startpos_board(), "c0c2");
        walk_and_diag(&a, Camp::Soldier, "A 中间变体和棋主线");
        // 场景 B: 标准开局炮第一步 c0c1 (判负着法, 兵方必胜)
        let b = game::apply_move(&game::startpos_board(), "c0c1");
        walk_and_diag(&b, Camp::Soldier, "B 标准开局炮c0c1后(兵胜)");
    }

    // 走完该着后炮方立即可吃的着法数(送子敞口)
    fn cannon_captures_after(board: &Board, mv: &str) -> usize {
        let child = game::apply_move(board, mv);
        game::legal_moves(&child, &Camp::Cannon).iter().filter(|m| is_cap(&child, m)).count()
    }

    fn walk_and_diag(start: &Board, first: Camp, label: &str) {
        let mut cur = *start;
        let mut camp = first;
        let (mut turns, mut chosen_enprise, mut with_safe_alt, mut free_sheep_lost, mut printed) = (0, 0, 0, 0, 0);
        for ply in 0..400 {
            if game::winner(&cur).is_some() {
                break;
            }
            if camp == Camp::Soldier {
                let Some(a) = analyze(&cur, camp) else { break };
                let chosen = a.pvs[0].clone();
                let val = state_value(&cur, camp).map(|v| cls_of(v, camp)).unwrap_or(0);
                let mut same_class = 0;
                let mut same_class_safe = 0;
                for mv in game::legal_moves(&cur, &camp) {
                    let Some(v) = move_value(&cur, camp, &mv) else { continue };
                    if cls_of(v, camp) == val {
                        same_class += 1;
                        if cannon_captures_after(&cur, &mv) == 0 {
                            same_class_safe += 1;
                        }
                    }
                }
                let child = game::apply_move(&cur, &chosen);
                let caps = cannon_captures_after(&cur, &chosen);
                let wolf_takes = analyze(&child, Camp::Cannon)
                    .map(|w| is_cap(&child, &w.pvs[0]))
                    .unwrap_or(false);
                turns += 1;
                if caps > 0 {
                    chosen_enprise += 1;
                    if same_class_safe > 0 {
                        with_safe_alt += 1;
                    }
                    if wolf_takes {
                        free_sheep_lost += 1;
                    }
                    if printed < 10 {
                        println!(
                            "{label} ply{ply} 值类={val} 推荐 {chosen} 敞口x{caps} 狼最佳回应是否吃={wolf_takes} 同类{same_class}个(不送子{same_class_safe}个) 分{}",
                            a.score
                        );
                        printed += 1;
                    }
                }
                cur = child;
            } else {
                match analyze(&cur, camp) {
                    Some(a) => cur = game::apply_move(&cur, &a.pvs[0]),
                    None => {
                        let mvs = game::legal_moves(&cur, &camp);
                        if mvs.is_empty() { break }
                        cur = game::apply_move(&cur, &mvs[0]);
                    }
                }
            }
            camp = camp.opposite();
            if ply % 50 == 49 {
                let (_, s) = game::counts(&cur);
                println!("{label} 时间线 ply{ply} 剩兵{s}");
            }
        }
        let (c, s) = game::counts(&cur);
        println!(
            "{label}: 兵方回合{turns} 推荐送子敞口{chosen_enprise}次(其中{with_safe_alt}次有同类不送子替代, {free_sheep_lost}次狼最优回应就是白吃) 终局剩炮{c}兵{s} {:?}",
            game::winner(&cur).map(|w| w.to_char())
        );
    }

    // 需要 k=6 层表已生成(release 运行): 验证查表分析可用且主线步步合法
    #[test]
    #[ignore]
    fn experiment_oracle_k6() {
        if std::env::var("APPDATA").is_err() {
            return;
        }
        let mut board = [[game::EMPTY; game::SIZE]; game::SIZE];
        board[0][1] = game::CANNON;
        board[0][2] = game::CANNON;
        board[0][3] = game::CANNON;
        for x in 0..5 {
            board[3][x] = game::SOLDIER;
        }
        board[4][0] = game::SOLDIER;
        let (cannons, soldiers) = game::counts(&board);
        assert_eq!((cannons, soldiers), (3, 6));

        let Some(a) = analyze(&board, Camp::Cannon) else {
            println!("oracle 不可用(k=6 表缺失), 跳过");
            return;
        };
        assert!(game::is_legal(&board, &a.pvs[0]), "最佳着法必须合法");
        // 主线在滚动棋盘上逐步验证合法性, 直到终局/截断
        let mut cur = board;
        let mut camp = Camp::Cannon;
        for mv in a.pvs.iter() {
            assert!(game::is_legal(&cur, mv), "主线着法 {mv} 非法");
            cur = game::apply_move(&cur, mv);
            camp = camp.opposite();
        }
        let val = state_value(&board, Camp::Cannon).unwrap();
        let cls = cls_of(val, Camp::Cannon);
        assert_eq!(cls, if a.score > 29000 { 1 } else if a.score < -29000 { -1 } else { 0 });
        println!(
            "oracle k=6: 值={} 分数={} 主线{}步 合法性通过",
            match val {
                VAL_WOLF => "炮胜",
                VAL_SHEEP => "兵胜",
                _ => "和",
            },
            a.score,
            a.pvs.len()
        );
    }
}
