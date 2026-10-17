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

// 最优着法: 值类优先, 同类吃子优先(加速终局并重置无吃子计数)
fn pick_best(board: &Board, camp: Camp) -> Option<(String, i32)> {
    let mut best: Option<(String, i32, bool)> = None;
    for mv in game::legal_moves(board, &camp) {
        let Some(v) = move_value(board, camp, &mv) else { continue };
        let cls = cls_of(v, camp);
        let cap = is_cap(board, &mv);
        let replace = match &best {
            None => true,
            Some((_, bc, bcap)) => cls > *bc || (cls == *bc && cap && !*bcap),
        };
        if replace {
            best = Some((mv, cls, cap));
        }
    }
    best.map(|(mv, cls, _)| (mv, cls))
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

    let scored: Vec<(String, i32)> = game::legal_moves(board, &camp)
        .into_iter()
        .map(|mv| {
            // 子层表缺失时按和棋类兜底(完整表下不会发生)
            let cls = move_value(board, camp, &mv).map(|v| cls_of(v, camp)).unwrap_or(0);
            (mv, cls)
        })
        .collect();
    let (best_mv, best_cls) = scored.iter().max_by_key(|(mv, cls)| (*cls, is_cap(board, mv)))?.clone();

    // 主线: 双方沿完美着法推演到终局, 和棋线路遇重复局面即停
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
        let Some((mv, _)) = pick_best(&cur, cur_camp) else { break };
        pv.push(mv.clone());
        cur = game::apply_move(&cur, &mv);
        cur_camp = cur_camp.opposite();
    }

    // 次优: 与最优同值类的着法
    let mut alternatives = Vec::new();
    let mut alt_scores = Vec::new();
    for (mv, cls) in scored.iter().skip(0) {
        if *cls == best_cls && *mv != best_mv && alternatives.len() < 4 {
            alternatives.push(mv.clone());
            alt_scores.push(0);
        }
    }

    // 分数映射: 胜 = MATE - 主线步数; 负 = 对称; 和 = 0
    let score = match val {
        1 => ai::MATE - pv.len() as isize,
        -1 => -(ai::MATE - pv.len() as isize),
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
