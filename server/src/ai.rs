// 三炮十五兵内置 AI: negamax + alpha-beta + 迭代加深, 替代原项目的外部 UCI 引擎
// 评分约定与原项目一致: 行棋方视角, 正=行棋方占优, 杀棋用 30000-n 编码
use std::time::Instant;

use crate::game::{self, Board, Camp};

pub const MATE: isize = 30000;

#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    pub depth: usize,  // 最大搜索深度
    pub time_ms: u64,  // 时间上限(毫秒)
}

// 提示档位 0-5 映射到固定深度: 0=跟随配置, 档位越高思考越浅(降智提示)
pub fn hint_depth(level: i32, configured: usize) -> usize {
    match level {
        1 => 2,
        2 => 3,
        3 => 4,
        4 => 5,
        5 => 6,
        _ => configured,
    }
    .max(1)
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub depth: usize,
    pub score: isize,           // 行棋方视角
    pub time_ms: u64,
    pub pvs: Vec<String>,       // 最佳主线(iccs)
    pub alternatives: Vec<String>, // 次优首着(iccs)
    pub alt_scores: Vec<isize>, // 次优与最优的分差(正数=落后多少)
}

struct SearchCtx {
    deadline: Option<Instant>,
    nodes: usize,
    aborted: bool,
}

impl SearchCtx {
    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.aborted {
            return false;
        }
        if self.nodes % 2048 == 0
            && let Some(deadline) = self.deadline
            && Instant::now() > deadline
        {
            self.aborted = true;
        }
        !self.aborted
    }
}

// 对当前局面做完整分析: 根节点逐着评分得到最优/次优与主线
pub fn analyze(board: &Board, camp: Camp, limits: SearchLimits, alt_gap: isize) -> Option<Analysis> {
    let start = Instant::now();
    let root_moves = game::legal_moves(board, &camp);
    if root_moves.is_empty() {
        return None; // 终局无着可走
    }
    let deadline = (limits.time_ms > 0).then(|| start + std::time::Duration::from_millis(limits.time_ms));
    let mut ctx = SearchCtx { deadline, nodes: 0, aborted: false };

    // 迭代加深: 上一层得分用于本层排序, 超时丢弃未完成的一层
    let mut ordered: Vec<String> = root_moves.clone();
    let mut scored: Vec<(String, isize, Vec<String>)> = Vec::new();
    let mut reached = 0usize;

    for depth in 1..=limits.depth {
        let mut round: Vec<(String, isize, Vec<String>)> = Vec::with_capacity(ordered.len());
        for mv in &ordered {
            let child = game::apply_move(board, mv);
            let mut tail = Vec::new();
            let score = -negamax(&child, camp.opposite(), depth - 1, -MATE * 2, MATE * 2, 1, &mut ctx, &mut tail);
            if ctx.aborted {
                break;
            }
            // 主线 = 根着法 + 子节点最优续着
            let mut pv = vec![mv.clone()];
            pv.extend(tail);
            round.push((mv.clone(), score, pv));
        }
        if ctx.aborted {
            break;
        }
        // 按得分降序排列, 作为下一层排序与最终结果
        round.sort_by(|a, b| b.1.cmp(&a.1));
        ordered = round.iter().map(|(mv, _, _)| mv.clone()).collect();
        scored = round;
        reached = depth;
        if let Some(best) = scored.first() {
            if best.1.abs() >= MATE - 100 {
                break; // 已找到杀棋, 无需更深
            }
        }
    }

    let (best_mv, best_score, best_pv) = scored.first()?.clone();
    if best_pv.is_empty() {
        return Some(Analysis {
            depth: reached,
            score: best_score,
            time_ms: start.elapsed().as_millis() as u64,
            pvs: vec![best_mv],
            alternatives: Vec::new(),
            alt_scores: Vec::new(),
        });
    }
    // 次优着法: 与最优分差在阈值内的其余首着
    let mut alternatives = Vec::new();
    let mut alt_scores = Vec::new();
    for (mv, score, _) in scored.iter().skip(1) {
        let gap = best_score - score;
        if gap <= alt_gap && alternatives.len() < 4 {
            alternatives.push(mv.clone());
            alt_scores.push(gap.max(0));
        }
    }
    Some(Analysis {
        depth: reached,
        score: best_score,
        time_ms: start.elapsed().as_millis() as u64,
        pvs: best_pv,
        alternatives,
        alt_scores,
    })
}

fn negamax(
    board: &Board,
    camp: Camp,
    depth: usize,
    mut alpha: isize,
    beta: isize,
    ply: usize,
    ctx: &mut SearchCtx,
    pv_out: &mut Vec<String>,
) -> isize {
    if !ctx.tick() {
        return eval(board, camp); // 超时: 用静态分冒充, 该层结果会被丢弃
    }
    // 终局(静止判定: 兵过少或一方无路可走)
    if let Some(w) = game::winner(board) {
        let mate = MATE - ply as isize;
        return if w == camp { mate } else { -mate };
    }
    if depth == 0 {
        return eval(board, camp);
    }
    let mut moves = game::legal_moves(board, &camp);
    // 吃子优先搜索, 提升剪枝效率
    moves.sort_by_key(|mv| {
        let to = &mv[2..];
        let occupied = game::xy_of(to).map(|(x, y)| board[y][x] != ' ').unwrap_or(false);
        if occupied { 0 } else { 1 }
    });

    let mut best = isize::MIN;
    let mut best_pv: Vec<String> = Vec::new();
    for mv in moves {
        let child = game::apply_move(board, &mv);
        let mut child_pv = Vec::new();
        let score = -negamax(&child, camp.opposite(), depth - 1, -beta, -alpha, ply + 1, ctx, &mut child_pv);
        if ctx.aborted {
            return best.max(score);
        }
        if score > best {
            best = score;
            best_pv = vec![mv.clone()];
            best_pv.extend(child_pv);
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    *pv_out = best_pv;
    best
}

// 局面静态评估: 正=利好炮方
fn eval_raw(board: &Board) -> isize {
    let (_, soldiers) = game::counts(board);
    let mut score = (15 - soldiers) as isize * 130; // 每吃一兵价值很高

    let mut cannon_cells = Vec::new();
    for y in 0..game::SIZE {
        for x in 0..game::SIZE {
            if board[y][x] == game::CANNON {
                cannon_cells.push((x, y));
            }
        }
    }
    let mut liberty = 0isize;
    let mut trapped = 0isize;
    for &(x, y) in &cannon_cells {
        let mv = game::piece_moves(board, x, y);
        let captures = mv
            .iter()
            .filter(|m| game::xy_of(&m[2..]).map(|(tx, ty)| board[ty][tx] == game::SOLDIER).unwrap_or(false))
            .count();
        liberty += mv.len() as isize + captures as isize; // 能吃的路更值钱
        if mv.is_empty() {
            trapped += 1;
        }
    }
    score += liberty * 9;
    if trapped == 3 {
        score -= 8000; // 接近兵胜(围死)
    } else if trapped > 0 {
        score -= trapped * 400;
    }

    // 兵逼近炮利于围堵: 距离越近兵方越好
    for y in 0..game::SIZE {
        for x in 0..game::SIZE {
            if board[y][x] == game::SOLDIER {
                if let Some(dist) = cannon_cells
                    .iter()
                    .map(|&(cx, cy)| cx.abs_diff(x) + cy.abs_diff(y))
                    .min()
                {
                    score -= (6 - dist as isize).max(0) * 7;
                }
            }
        }
    }
    score
}

// 行棋方视角评分
fn eval(board: &Board, camp: Camp) -> isize {
    let raw = eval_raw(board);
    if camp == Camp::Cannon {
        raw
    } else {
        -raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{startpos_board, Camp};

    // 实验工具(默认跳过): cargo test -- --ignored 运行
    // 深度分析两种开局, 观察行棋方(炮)视角评分是否系统性偏向兵方
    #[test]
    #[ignore]
    fn experiment_opening_analysis() {
        for (name, board) in [
            ("标准开局a/c/e", startpos_board()),
            ("中间变体b/c/d", crate::game::middle_startpos_board()),
        ] {
            let r = analyze(&board, Camp::Cannon, SearchLimits { depth: 12, time_ms: 20_000 }, 100).unwrap();
            println!(
                "{name}: 深度{} 炮方视角分数={} 最佳={:?} 主线={:?}",
                r.depth, r.score, r.pvs.first(), r.pvs
            );
        }
    }

    // 自对弈统计: 每局先随机走 2*4 步制造多样局面, 再引擎对弈(同深度), 统计胜率
    #[test]
    #[ignore]
    fn experiment_selfplay_tally() {
        let games = 12usize;
        let (mut cannon_wins, mut soldier_wins, mut draws, mut total_plies) = (0, 0, 0, 0usize);
        for g in 1..=games {
            let mut board = crate::game::middle_startpos_board();
            let mut camp = Camp::Cannon;
            // 简易 LCG 随机前缀: 每局不同的乱开局
            let mut rng: u64 = 0x9E3779B97F4A7C15u64.wrapping_mul(g as u64);
            let mut next = move || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 33) as usize
            };
            let mut plies = 0usize;
            let mut no_capture = 0usize;
            let outcome = loop {
                let moves = crate::game::legal_moves(&board, &camp);
                if moves.is_empty() {
                    break "无着";
                }
                let mv = if plies < 8 {
                    moves[next() % moves.len()].clone()
                } else {
                    let Some(result) = analyze(&board, camp, SearchLimits { depth: 5, time_ms: 300 }, 50) else {
                        break "无着";
                    };
                    result.pvs[0].clone()
                };
                let to = mv[2..].to_string();
                if let Some((x, y)) = crate::game::xy_of(&to)
                    && board[y][x] != crate::game::EMPTY
                {
                    no_capture = 0;
                } else {
                    no_capture += 1;
                }
                board = crate::game::apply_move(&board, &mv);
                plies += 1;
                if let Some(winner) = crate::game::winner(&board) {
                    break if winner == Camp::Cannon { "炮胜" } else { "兵胜" };
                }
                if no_capture >= crate::game::DRAW_NO_CAPTURE_PLIES {
                    break "和棋";
                }
                camp = camp.opposite();
            };
            match outcome {
                "炮胜" => cannon_wins += 1,
                "兵胜" => soldier_wins += 1,
                _ => draws += 1,
            }
            total_plies += plies;
            let (_, soldiers) = crate::game::counts(&board);
            println!("第{g}局: {outcome} 步数={plies} 剩兵={soldiers}");
        }
        println!(
            "自对弈{games}局: 炮胜{cannon_wins} 兵胜{soldier_wins} 和{draws} 平均步数{:.0}",
            total_plies as f64 / games as f64
        );
    }

    #[test]
    fn test_analyze_startpos() {
        let board = startpos_board();
        let result = analyze(&board, Camp::Cannon, SearchLimits { depth: 4, time_ms: 1000 }, 100).unwrap();
        assert!(result.pvs.len() >= 1);
        assert!(game::is_legal(&board, &result.pvs[0]), "建议着法必须合法");
        assert_eq!(result.depth, 4);
    }

    #[test]
    fn test_find_trap_mate() {
        // 一步围毙: 兵走 c2→c1 封死最后一门可动的炮
        let mut board = [[game::EMPTY; game::SIZE]; game::SIZE];
        board[0] = [game::CANNON, game::SOLDIER, game::CANNON, game::SOLDIER, game::CANNON];
        board[1] = [game::SOLDIER, game::EMPTY, game::EMPTY, game::EMPTY, game::SOLDIER];
        board[2] = [game::EMPTY, game::EMPTY, game::SOLDIER, game::EMPTY, game::EMPTY];
        let result = analyze(&board, Camp::Soldier, SearchLimits { depth: 6, time_ms: 3000 }, 100).unwrap();
        assert_eq!(result.pvs[0], "c2c1", "应找到封炮的关键着法");
        assert!(result.score > 29000, "一步围毙应接近杀棋分, 实际: {}", result.score);
    }

    #[test]
    fn test_selfplay_terminates() {
        // 双方深度4自对弈, 应在限时内分出胜负或达到步数上限
        let mut board = startpos_board();
        let mut camp = Camp::Cannon;
        let start = Instant::now();
        let mut plies = 0;
        let mut no_capture = 0;
        loop {
            if let Some(result) = analyze(&board, camp, SearchLimits { depth: 4, time_ms: 500 }, 50) {
                let mv = result.pvs[0].clone();
                let to = mv[2..].to_string();
                if let Some((x, y)) = game::xy_of(&to) {
                    no_capture = if board[y][x] != game::EMPTY { 0 } else { no_capture + 1 };
                }
                board = game::apply_move(&board, &mv);
                plies += 1;
            } else {
                break; // 无着可走, 终局
            }
            if game::winner(&board).is_some() || no_capture >= game::DRAW_NO_CAPTURE_PLIES {
                break;
            }
            camp = camp.opposite();
            assert!(plies < 400, "自对弈步数异常");
        }
        assert!(start.elapsed().as_secs() < 30, "自对弈过慢");
        assert!(game::winner(&board).is_some() || plies >= game::DRAW_NO_CAPTURE_PLIES);
    }
}
