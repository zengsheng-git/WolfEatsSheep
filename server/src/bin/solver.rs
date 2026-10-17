// 全量求解 CLI: 逐层计算 k=3..15 的局面值表并写盘, 支持断点续跑与开局验证
// 用法: cargo run --release --bin solver [-- <输出目录>]
use std::path::PathBuf;
use std::time::Instant;

use sanpao_lib::game::{self, Camp};
use sanpao_lib::oracle;
use sanpao_lib::solver::{self, VAL_DRAW};

fn default_out() -> PathBuf {
    if let Some(p) = std::env::args().nth(1) {
        return PathBuf::from(p);
    }
    if let Ok(appdata) = std::env::var("APPDATA") {
        let dir = PathBuf::from(appdata).join("sanpao").join("tables");
        if std::fs::create_dir_all(&dir).is_ok() {
            return dir;
        }
    }
    PathBuf::from("tables")
}

fn main() {
    let out = default_out();
    println!("输出目录: {}", out.display());
    let mut prev: Option<Vec<u8>> = None;
    for k in 3..=15usize {
        let path = out.join(solver::layer_filename(k));
        if let Some(t) = solver::read_table(&out, k) {
            println!("k={k}: 已存在, 跳过 ({} 字节)", t.len());
            prev = Some(t);
            continue;
        }
        let t0 = Instant::now();
        let table = solver::solve_layer(k, prev.take());
        let (w, s, d) = solver::table_stats(&table);
        let (per_side, _) = solver::layer_counts(k);
        match solver::write_table(&out, k, &table) {
            Ok(()) => println!(
                "k={k}: 炮胜{w} 兵胜{s} 和{d} / 共{} 局面, 用时{:.1}s, 已写入 {}",
                per_side * 2,
                t0.elapsed().as_secs_f64(),
                path.display()
            ),
            Err(e) => {
                eprintln!("k={k}: 写盘失败: {e}");
                std::process::exit(1);
            }
        }
        prev = Some(table);
    }

    // 开局验证: 对照 0irina/lang-chi-yang-qi 的破解结论
    // (中间变体开局为和棋, 唯一不亏着法为中炮跳吃 c0c2)
    // safety: 单线程 CLI 中设置进程级环境变量
    unsafe { std::env::set_var("SANPAO_TABLES", &out) };
    for (name, board) in [
        ("中间变体b/c/d", game::middle_startpos_board()),
        ("标准开局a/c/e", game::startpos_board()),
    ] {
        let Some(v) = oracle::state_value(&board, Camp::Cannon) else {
            println!("{name}: 无法查询(表缺失)");
            continue;
        };
        let val_text = match v {
            solver::VAL_WOLF => "炮胜",
            solver::VAL_SHEEP => "兵胜",
            VAL_DRAW => "和",
            _ => "?",
        };
        println!("{name}: 表值 = {val_text}");
        for mv in game::legal_moves(&board, &Camp::Cannon) {
            let child = game::apply_move(&board, &mv);
            let Some(cv) = oracle::state_value(&child, Camp::Soldier) else { continue };
            let cv_text = match cv {
                solver::VAL_WOLF => "炮胜",
                solver::VAL_SHEEP => "兵胜",
                VAL_DRAW => "和",
                _ => "?",
            };
            println!("  着法 {mv} → {cv_text}");
        }
    }

    // oracle 端到端抽查: 走一段固定线路到中局, 验证完美分析可用且着法合法
    if let Some((name, board)) = [("中间变体b/c/d", game::middle_startpos_board())].into_iter().next() {
        let mut cur = board;
        let mut camp = Camp::Cannon;
        for mv in ["c0c2", "d2d1", "b0b2", "e2e1", "b2b1", "b3b2", "c2e2", "c3c2"] {
            if game::is_legal(&cur, mv) {
                cur = game::apply_move(&cur, mv);
                camp = camp.opposite();
            }
        }
        match oracle::analyze(&cur, camp) {
            Some(a) => println!(
                "{name} 8步后 oracle: 最佳={:?} 分数={} 主线长={} 合法={}",
                a.pvs.first(),
                a.score,
                a.pvs.len(),
                a.pvs.first().map(|m| game::is_legal(&cur, m)).unwrap_or(false)
            ),
            None => println!("{name} 8步后 oracle: 不可用(对应层表缺失)"),
        }
        let _ = name;
    }
}
