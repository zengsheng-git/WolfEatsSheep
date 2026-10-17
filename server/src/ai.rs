// 三炮十五兵内置 AI: negamax + alpha-beta + 迭代加深, 配套置换表 / 静态搜索 / 杀手历史启发
// / PVS + LMR / 路径重复判和, 替代原项目的外部 UCI 引擎
// 评分约定与原项目一致: 行棋方视角, 正=行棋方占优, 杀棋用 30000-n 编码
use std::time::Instant;

use crate::game::{self, Board, Camp};

pub const MATE: isize = 30000;
const MATE_BOUND: isize = MATE - 1000; // 超过视为杀棋分, 置换表存取需换算成局面相对距离
const TT_BITS: usize = 20;
const TT_LEN: usize = 1 << TT_BITS;
const MAX_PLY: usize = 128;

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

// ---- 置换表 ----
// 定长直接索引, key 校验命中; 项内分数按"局面相对杀棋距离"存储(与根距离解耦)
#[derive(Clone, Copy, Default)]
struct TTEntry {
    key: u64,
    mv: u16,   // 0=无着法(紧凑编码, 见 mv_encode)
    score: i16,
    depth: u8,
    flag: u8,  // 0=精确 1=下界(≥) 2=上界(≤)
}

fn to_tt_score(score: isize, ply: usize) -> isize {
    if score >= MATE_BOUND {
        score + ply as isize
    } else if score <= -MATE_BOUND {
        score - ply as isize
    } else {
        score
    }
}

fn from_tt_score(score: isize, ply: usize) -> isize {
    if score >= MATE_BOUND {
        score - ply as isize
    } else if score <= -MATE_BOUND {
        score + ply as isize
    } else {
        score
    }
}

// 着法紧凑编码: from*25+to, "a0a0" 不可能是合法着法, 0 可作哨兵
fn mv_encode(mv: &str) -> u16 {
    match (game::xy_of(&mv[..2]), game::xy_of(&mv[2..])) {
        (Some((fx, fy)), Some((tx, ty))) => ((fy * game::SIZE + fx) * 25 + ty * game::SIZE + tx) as u16,
        _ => 0,
    }
}

fn mv_decode(m: u16) -> Option<String> {
    if m == 0 {
        return None;
    }
    let f = (m / 25) as usize;
    let t = (m % 25) as usize;
    if f >= 25 || t >= 25 {
        return None;
    }
    Some(game::pos_of(f % game::SIZE, f / game::SIZE) + &game::pos_of(t % game::SIZE, t / game::SIZE))
}

fn is_capture(board: &Board, mv: &str) -> bool {
    game::xy_of(&mv[2..]).map(|(x, y)| board[y][x] != game::EMPTY).unwrap_or(false)
}

struct Engine {
    zobrist: [[u64; 2]; 25], // [格][0=炮 1=兵]
    tt: Vec<TTEntry>,
    killers: [[u16; 2]; MAX_PLY],
    history: [[i32; 25]; 25], // [from格][to格]
    ctx: SearchCtx,
    path: Vec<u64>, // 当前搜索路径哈希, 用于重复局面判和
}

impl Engine {
    fn new(ctx: SearchCtx) -> Self {
        let mut zobrist = [[0u64; 2]; 25];
        let mut s = 0x853C_49E6_748F_EA9Bu64;
        for cell in zobrist.iter_mut() {
            for v in cell.iter_mut() {
                s = s.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
                *v = s;
            }
        }
        Engine {
            zobrist,
            tt: vec![TTEntry::default(); TT_LEN],
            killers: [[0; 2]; MAX_PLY],
            history: [[0; 25]; 25],
            ctx,
            path: Vec::new(),
        }
    }

    fn hash(&self, board: &Board) -> u64 {
        let mut h = 0u64;
        for y in 0..game::SIZE {
            for x in 0..game::SIZE {
                match board[y][x] {
                    game::CANNON => h ^= self.zobrist[y * game::SIZE + x][0],
                    game::SOLDIER => h ^= self.zobrist[y * game::SIZE + x][1],
                    _ => {}
                }
            }
        }
        h
    }

    fn tt_move(&self, h: u64) -> Option<u16> {
        let e = &self.tt[(h as usize) & (TT_LEN - 1)];
        (e.key == h && e.mv != 0).then_some(e.mv)
    }

    fn tt_probe(&self, h: u64, depth: usize, alpha: isize, beta: isize, ply: usize) -> Option<isize> {
        let e = &self.tt[(h as usize) & (TT_LEN - 1)];
        if e.key != h || (e.depth as usize) < depth {
            return None;
        }
        let score = from_tt_score(e.score as isize, ply);
        match e.flag {
            0 => Some(score),
            1 if score >= beta => Some(score),
            2 if score <= alpha => Some(score),
            _ => None,
        }
    }

    fn tt_store(&mut self, h: u64, depth: usize, flag: u8, score: isize, mv: u16, ply: usize) {
        let idx = (h as usize) & (TT_LEN - 1);
        let e = &self.tt[idx];
        if e.key == h && e.flag == 0 && e.depth as usize > depth {
            return; // 已有更深的精确项, 保留
        }
        let s = to_tt_score(score, ply);
        self.tt[idx] = TTEntry {
            key: h,
            mv,
            score: s as i16,
            depth: depth.min(255) as u8,
            flag,
        };
    }

    fn update_quiet_stats(&mut self, enc: u16, ply: usize, depth: usize) {
        let k = ply.min(MAX_PLY - 1);
        let prev = self.killers[k];
        if prev[0] != enc {
            self.killers[k] = [enc, prev[0]];
        }
        let from = (enc / 25) as usize;
        let to = (enc % 25) as usize;
        if from < 25 && to < 25 {
            // 封顶防止历史分增长到干扰吃子/杀手优先级的高度
            self.history[from][to] = (self.history[from][to] + (depth * depth) as i32).min(1 << 16);
        }
    }

    fn negamax(&mut self, board: &Board, camp: Camp, depth: usize, mut alpha: isize, beta: isize, ply: usize) -> isize {
        if !self.ctx.tick() {
            return 0; // 超时: 值无意义, 根节点会丢弃整层
        }
        if let Some(w) = winner_fast(board) {
            let mate = MATE - ply as isize;
            return if w == camp { mate } else { -mate };
        }
        if depth == 0 {
            return self.qsearch(board, camp, alpha, beta, ply);
        }
        let h = self.hash(board);
        // 路径 = 最近一次吃子以来的局面序列; 路径内重复或无吃子步数耗尽都按和棋处理
        if self.path.contains(&h) {
            return 0;
        }
        if self.path.len() >= game::DRAW_NO_CAPTURE_PLIES {
            return 0;
        }
        if let Some(score) = self.tt_probe(h, depth, alpha, beta, ply) {
            return score;
        }
        let tt_mv = self.tt_move(h);

        let mut ordered: Vec<(isize, String)> = game::legal_moves(board, &camp)
            .into_iter()
            .map(|mv| {
                let enc = mv_encode(&mv);
                let mut s = if is_capture(board, &mv) {
                    1 << 19
                } else {
                    let k = ply.min(MAX_PLY - 1);
                    if self.killers[k][0] == enc {
                        1 << 18
                    } else if self.killers[k][1] == enc {
                        1 << 17
                    } else if let (Some((fx, fy)), Some((tx, ty))) = (game::xy_of(&mv[..2]), game::xy_of(&mv[2..])) {
                        self.history[fy * game::SIZE + fx][ty * game::SIZE + tx].max(0) as isize
                    } else {
                        0
                    }
                };
                if Some(enc) == tt_mv {
                    s += 1 << 20;
                }
                (s, mv)
            })
            .collect();
        ordered.sort_by(|a, b| b.0.cmp(&a.0));

        let orig_alpha = alpha;
        let mut best = isize::MIN;
        let mut best_enc: u16 = 0;
        self.path.push(h);
        for (i, (_, mv)) in ordered.iter().enumerate() {
            let cap = is_capture(board, mv);
            let child = game::apply_move(board, mv);
            // 吃子清零无吃子计数: 重复判定路径随之重开(吃过子后的"重复"并不违反规则)
            let saved_path = if cap { Some(std::mem::take(&mut self.path)) } else { None };
            let mut score;
            if i == 0 {
                score = -self.negamax(&child, camp.opposite(), depth - 1, -beta, -alpha, ply + 1);
            } else {
                // PVS: 零窗口试探; 静着法晚搜降层(LMR), 需要时逐步升回
                let red = if !cap && i >= 4 && depth >= 3 { 2 } else { 1 };
                score = -self.negamax(&child, camp.opposite(), depth - red, -alpha - 1, -alpha, ply + 1);
                if score > alpha && red > 1 {
                    score = -self.negamax(&child, camp.opposite(), depth - 1, -alpha - 1, -alpha, ply + 1);
                }
                if score > alpha && score < beta {
                    score = -self.negamax(&child, camp.opposite(), depth - 1, -beta, -alpha, ply + 1);
                }
            }
            if let Some(p) = saved_path {
                self.path = p;
            }
            if self.ctx.aborted {
                break;
            }
            if score > best {
                best = score;
                best_enc = mv_encode(mv);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                if !cap {
                    self.update_quiet_stats(mv_encode(mv), ply, depth);
                }
                break;
            }
        }
        self.path.pop();
        if self.ctx.aborted || best == isize::MIN {
            return 0;
        }
        let flag = if best <= orig_alpha { 2 } else if best >= beta { 1 } else { 0 };
        self.tt_store(h, depth, flag, best, best_enc, ply);
        best
    }

    // 静态搜索: 只延伸炮的吃子(本游戏全部战术), 消除"深度用尽时正被跳吃"的视界效应
    fn qsearch(&mut self, board: &Board, camp: Camp, mut alpha: isize, beta: isize, ply: usize) -> isize {
        if !self.ctx.tick() {
            return 0;
        }
        if let Some(w) = winner_fast(board) {
            let mate = MATE - ply as isize;
            return if w == camp { mate } else { -mate };
        }
        let stand = eval(board, camp);
        if camp != Camp::Cannon || ply >= MAX_PLY {
            return stand; // 兵无吃子着法, 静态即终值
        }
        if stand >= beta {
            return stand;
        }
        if stand > alpha {
            alpha = stand;
        }
        let mut best = stand;
        'outer: for y in 0..game::SIZE {
            for x in 0..game::SIZE {
                if board[y][x] != game::CANNON {
                    continue;
                }
                // 直接枚举炮的隔空吃: 中间空 + 2 格外是兵, 免去整份着法生成
                for &(dx, dy) in &AI_DIRS {
                    let mx = x as i32 + dx;
                    let my = y as i32 + dy;
                    let tx = x as i32 + dx * 2;
                    let ty = y as i32 + dy * 2;
                    if !(in_bounds(mx) && in_bounds(my) && in_bounds(tx) && in_bounds(ty)) {
                        continue;
                    }
                    if board[my as usize][mx as usize] != game::EMPTY
                        || board[ty as usize][tx as usize] != game::SOLDIER
                    {
                        continue;
                    }
                    let mv = game::pos_of(x, y) + &game::pos_of(tx as usize, ty as usize);
                    let child = game::apply_move(board, &mv);
                    let score = -self.qsearch(&child, camp.opposite(), -beta, -alpha, ply + 1);
                    if self.ctx.aborted {
                        return 0;
                    }
                    if score > best {
                        best = score;
                        if score > alpha {
                            alpha = score;
                            if alpha >= beta {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        best
    }

    // 主线回溯: 沿置换表最佳着法走, 直到无项/非法/重复/终局
    fn pv_from_tt(&self, board: &Board, camp: Camp, max_len: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut b = *board;
        let mut c = camp;
        let mut seen = std::collections::HashSet::new();
        while out.len() < max_len {
            let h = self.hash(&b);
            if !seen.insert(h) {
                break;
            }
            let Some(mv) = self.tt_move(h).and_then(mv_decode) else {
                break;
            };
            if !game::is_legal(&b, &mv) {
                break;
            }
            b = game::apply_move(&b, &mv);
            out.push(mv);
            if game::winner(&b).is_some() {
                break;
            }
            c = c.opposite();
        }
        out
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
    let ctx = SearchCtx { deadline, nodes: 0, aborted: false };
    let mut engine = Engine::new(ctx);
    engine.path.push(engine.hash(board));

    // 迭代加深: 上一层得分用于本层排序, 超时丢弃未完成的一层
    let mut ordered: Vec<String> = root_moves.clone();
    let mut scored: Vec<(String, isize)> = Vec::new();
    let mut reached = 0usize;

    for depth in 1..=limits.depth {
        let mut round: Vec<(String, isize)> = Vec::with_capacity(ordered.len());
        for mv in &ordered {
            let child = game::apply_move(board, mv);
            // 根上用全窗口保证次优分差精确(置换表只可能回吐精确项)
            let score = -engine.negamax(&child, camp.opposite(), depth - 1, -MATE * 2, MATE * 2, 1);
            if engine.ctx.aborted {
                break;
            }
            round.push((mv.clone(), score));
        }
        if engine.ctx.aborted {
            break;
        }
        round.sort_by(|a, b| b.1.cmp(&a.1));
        ordered = round.iter().map(|(mv, _)| mv.clone()).collect();
        scored = round;
        reached = depth;
        if scored.first().map(|(_, s)| s.abs() >= MATE - 100).unwrap_or(false) {
            break; // 已找到杀棋, 无需更深
        }
    }

    let (best_mv, best_score) = scored.first()?.clone();
    // 主线沿置换表回溯(子树命中置换表时不再收集 pv, 统一从这里重建)
    let child = game::apply_move(board, &best_mv);
    let mut pvs = vec![best_mv.clone()];
    pvs.extend(engine.pv_from_tt(&child, camp.opposite(), 23));

    // 次优着法: 与最优分差在阈值内的其余首着
    let mut alternatives = Vec::new();
    let mut alt_scores = Vec::new();
    for (mv, score) in scored.iter().skip(1) {
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
        pvs,
        alternatives,
        alt_scores,
    })
}

const AI_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

fn in_bounds(v: i32) -> bool {
    (0..game::SIZE as i32).contains(&v)
}

// 该子是否存在任何着法。炮的跳吃必须隔空格, 有跳吃时该空格本身也是平移着法,
// 因此炮/兵都等价于"存在相邻空点", 不做字符串分配
fn piece_can_move(board: &Board, x: usize, y: usize) -> bool {
    for &(dx, dy) in &AI_DIRS {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if in_bounds(nx) && in_bounds(ny) && board[ny as usize][nx as usize] == game::EMPTY {
            return true;
        }
    }
    false
}

// winner() 的快速版: 一次扫描同时数子与探双方可动性, 避免每节点两次完整着法生成
fn winner_fast(board: &Board) -> Option<Camp> {
    let mut soldiers = 0usize;
    let mut cannon_can_move = false;
    let mut soldier_can_move = false;
    for y in 0..game::SIZE {
        for x in 0..game::SIZE {
            match board[y][x] {
                game::CANNON => {
                    if !cannon_can_move && piece_can_move(board, x, y) {
                        cannon_can_move = true;
                    }
                }
                game::SOLDIER => {
                    soldiers += 1;
                    if !soldier_can_move && piece_can_move(board, x, y) {
                        soldier_can_move = true;
                    }
                }
                _ => {}
            }
        }
    }
    if soldiers <= game::SOLDIER_LOSE_COUNT {
        Some(Camp::Cannon)
    } else if !cannon_can_move {
        Some(Camp::Soldier)
    } else if !soldier_can_move {
        Some(Camp::Cannon)
    } else {
        None
    }
}

// 局面静态评估: 正=利好炮方
// 威胁项带行棋方权重: 轮到炮走时"下步即可跳吃"的兵接近已到手, 轮到兵走时只是隐患
fn eval_raw(board: &Board, camp: Camp) -> isize {
    let (_, soldiers) = game::counts(board);
    let mut score = (15 - soldiers) as isize * 130; // 每吃一兵价值很高
    if soldiers < 6 {
        score += (6 - soldiers) as isize * 40; // 兵越少越接近"剩 3 即胜", 每兵加价
    }

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
        // 直接扫四邻: 空邻可平移, 空邻 + 2 格外有兵则可跳吃(计数翻倍, 见 liberty 权重)
        let mut quiet = 0isize;
        let mut caps = 0isize;
        for &(dx, dy) in &AI_DIRS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if !in_bounds(nx) || !in_bounds(ny) {
                continue;
            }
            if board[ny as usize][nx as usize] != game::EMPTY {
                continue;
            }
            quiet += 1;
            let tx = x as i32 + dx * 2;
            let ty = y as i32 + dy * 2;
            if in_bounds(tx) && in_bounds(ty) && board[ty as usize][tx as usize] == game::SOLDIER {
                caps += 1;
            }
        }
        liberty += quiet + caps * 2; // 能吃的路更值钱
        if quiet + caps == 0 {
            trapped += 1;
        }
    }
    score += liberty * 9;
    if trapped == 3 {
        score -= 8000; // 接近兵胜(围死)
    } else if trapped > 0 {
        score -= trapped * 400;
    }

    // 威胁: 兵处在炮的射击线上(隔一空格), 轮炮走即可吃
    let mut threatened = 0isize;
    for sy in 0..game::SIZE {
        'soldier: for sx in 0..game::SIZE {
            if board[sy][sx] != game::SOLDIER {
                continue;
            }
            for &(dx, dy) in &AI_DIRS {
                let mx = sx as i32 + dx;
                let my = sy as i32 + dy;
                let cx = sx as i32 + dx * 2;
                let cy = sy as i32 + dy * 2;
                if (0..game::SIZE as i32).contains(&mx)
                    && (0..game::SIZE as i32).contains(&my)
                    && (0..game::SIZE as i32).contains(&cx)
                    && (0..game::SIZE as i32).contains(&cy)
                    && board[my as usize][mx as usize] == game::EMPTY
                    && board[cy as usize][cx as usize] == game::CANNON
                {
                    threatened += 1;
                    continue 'soldier;
                }
            }
        }
    }
    let threat_weight = if camp == Camp::Cannon { 110 } else { 55 };
    score += threatened * threat_weight;

    // 兵逼近炮利于围堵: 距离越近兵方越好
    for y in 0..game::SIZE {
        for x in 0..game::SIZE {
            if board[y][x] == game::SOLDIER {
                if let Some(dist) = cannon_cells
                    .iter()
                    .map(|&(cx, cy)| cx.abs_diff(x) + cy.abs_diff(y))
                    .min()
                {
                    score -= (6 - dist as isize).max(0) * 12;
                }
            }
        }
    }
    score
}

// 行棋方视角评分
fn eval(board: &Board, camp: Camp) -> isize {
    let raw = eval_raw(board, camp);
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

    // ---- 旧版引擎副本(仅测试用): 重写前的裸 negamax + 旧评估, 作同资源对弈基准 ----
    mod legacy {
        use crate::game::{self, Board, Camp};
        use std::time::Instant;

        pub const MATE: isize = 30000;

        pub struct Limits {
            pub depth: usize,
            pub time_ms: u64,
        }

        struct Ctx {
            deadline: Option<Instant>,
            nodes: usize,
            aborted: bool,
        }

        impl Ctx {
            fn tick(&mut self) -> bool {
                self.nodes += 1;
                if self.aborted {
                    return false;
                }
                if self.nodes % 2048 == 0
                    && let Some(d) = self.deadline
                    && Instant::now() > d
                {
                    self.aborted = true;
                }
                !self.aborted
            }
        }

        pub fn analyze(board: &Board, camp: Camp, limits: Limits) -> Option<(String, usize)> {
            let start = Instant::now();
            let root_moves = game::legal_moves(board, &camp);
            if root_moves.is_empty() {
                return None;
            }
            let deadline = (limits.time_ms > 0).then(|| start + std::time::Duration::from_millis(limits.time_ms));
            let mut ctx = Ctx { deadline, nodes: 0, aborted: false };
            let mut ordered = root_moves;
            let mut best = String::new();
            let mut reached = 0usize;
            for depth in 1..=limits.depth {
                let mut round = Vec::new();
                for mv in &ordered {
                    let child = game::apply_move(board, mv);
                    let score = -negamax(&child, camp.opposite(), depth - 1, -MATE * 2, MATE * 2, 1, &mut ctx);
                    if ctx.aborted {
                        break;
                    }
                    round.push((score, mv.clone()));
                }
                if ctx.aborted {
                    break;
                }
                round.sort_by(|a, b| b.0.cmp(&a.0));
                if let Some((_, mv)) = round.first() {
                    best = mv.clone();
                    reached = depth;
                    if round[0].0.abs() >= MATE - 100 {
                        break;
                    }
                }
                ordered = round.into_iter().map(|(_, mv)| mv).collect();
            }
            (!best.is_empty()).then_some((best, reached))
        }

        fn negamax(
            board: &Board,
            camp: Camp,
            depth: usize,
            mut alpha: isize,
            beta: isize,
            ply: usize,
            ctx: &mut Ctx,
        ) -> isize {
            if !ctx.tick() {
                return legacy_eval(board, camp);
            }
            if let Some(w) = game::winner(board) {
                let mate = MATE - ply as isize;
                return if w == camp { mate } else { -mate };
            }
            if depth == 0 {
                return legacy_eval(board, camp);
            }
            let mut moves = game::legal_moves(board, &camp);
            moves.sort_by_key(|mv| {
                let to = &mv[2..];
                let occupied = game::xy_of(to).map(|(x, y)| board[y][x] != ' ').unwrap_or(false);
                if occupied { 0 } else { 1 }
            });
            let mut best = isize::MIN;
            for mv in moves {
                let child = game::apply_move(board, &mv);
                let score = -negamax(&child, camp.opposite(), depth - 1, -beta, -alpha, ply + 1, ctx);
                if ctx.aborted {
                    return best.max(score);
                }
                if score > best {
                    best = score;
                }
                if best > alpha {
                    alpha = best;
                }
                if alpha >= beta {
                    break;
                }
            }
            best
        }

        fn legacy_eval_raw(board: &Board) -> isize {
            let (_, soldiers) = game::counts(board);
            let mut score = (15 - soldiers) as isize * 130;
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
                liberty += mv.len() as isize + captures as isize;
                if mv.is_empty() {
                    trapped += 1;
                }
            }
            score += liberty * 9;
            if trapped == 3 {
                score -= 8000;
            } else if trapped > 0 {
                score -= trapped * 400;
            }
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

        fn legacy_eval(board: &Board, camp: Camp) -> isize {
            let raw = legacy_eval_raw(board);
            if camp == Camp::Cannon { raw } else { -raw }
        }
    }

    // 同资源对弈: 新引擎 vs 旧实现副本, 双方同深度同限时限, 各执炮兵交替
    #[test]
    #[ignore]
    fn experiment_match_vs_legacy() {
        let rounds = 6usize; // 每种执子 x 两种开局
        let (mut new_pts, mut old_pts) = (0f64, 0f64);
        let (mut new_depth_sum, mut old_depth_sum, mut depth_n) = (0usize, 0usize, 0usize);
        for g in 0..rounds * 2 {
            let new_plays_cannon = g % 2 == 0;
            let mut board = if (g / 2) % 2 == 0 {
                crate::game::middle_startpos_board()
            } else {
                startpos_board()
            };
            let mut camp = Camp::Cannon;
            let mut rng: u64 = 0x9E3779B97F4A7C15u64.wrapping_mul((g + 1) as u64);
            let mut plies = 0usize;
            let mut no_capture = 0usize;
            let outcome = loop {
                let moves = game::legal_moves(&board, &camp);
                if moves.is_empty() {
                    break "无着";
                }
                let mv = if plies < 6 {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    moves[((rng >> 33) as usize) % moves.len()].clone()
                } else if (camp == Camp::Cannon) == new_plays_cannon {
                    let r = analyze(&board, camp, SearchLimits { depth: 16, time_ms: 400 }, 50).unwrap();
                    new_depth_sum += r.depth;
                    depth_n += 1;
                    r.pvs[0].clone()
                } else {
                    let Some((mv, depth)) = legacy::analyze(&board, camp, legacy::Limits { depth: 16, time_ms: 400 }) else {
                        break "无着";
                    };
                    old_depth_sum += depth;
                    depth_n += 1;
                    mv
                };
                if let Some((x, y)) = game::xy_of(&mv[2..]) {
                    no_capture = if board[y][x] != game::EMPTY { 0 } else { no_capture + 1 };
                }
                board = game::apply_move(&board, &mv);
                plies += 1;
                if let Some(winner) = game::winner(&board) {
                    break if winner == Camp::Cannon { "炮胜" } else { "兵胜" };
                }
                if no_capture >= game::DRAW_NO_CAPTURE_PLIES {
                    break "和棋";
                }
                camp = camp.opposite();
            };
            let new_won = match outcome {
                "炮胜" => new_plays_cannon,
                "兵胜" => !new_plays_cannon,
                _ => false,
            };
            let draw = outcome == "和棋" || outcome == "无着";
            new_pts += if new_won { 1.0 } else if draw { 0.5 } else { 0.0 };
            old_pts += if !new_won && !draw { 1.0 } else if draw { 0.5 } else { 0.0 };
            println!(
                "第{}局: 新引擎执{} {} (步数={})",
                g + 1,
                if new_plays_cannon { "炮" } else { "兵" },
                if new_won { "胜" } else if draw { "和" } else { "负" },
                plies
            );
        }
        let total = (rounds * 2) as f64;
        println!(
            "新旧同资源对弈: 新引擎 {} : {} 旧引擎 (共{}局); 平均搜索深度 新={:.1} 旧={:.1}",
            new_pts,
            old_pts,
            total,
            new_depth_sum as f64 / depth_n as f64,
            old_depth_sum as f64 / depth_n as f64
        );
        assert!(new_pts > total / 2.0, "新引擎应胜过旧实现: {} : {}", new_pts, old_pts);
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
    fn test_middle_startpos_prefers_center_shot() {
        // 对照 0irina/lang-chi-yang-qi 的完全破解结论: 中间变体开局唯一不亏的
        // 着法是中炮跳吃(c0c2), 两侧炮跳吃与其余静着皆败
        let board = crate::game::middle_startpos_board();
        let result = analyze(&board, Camp::Cannon, SearchLimits { depth: 10, time_ms: 5000 }, 100).unwrap();
        assert_eq!(result.pvs[0], "c0c2", "开局应选中炮跳吃(破解唯一正招), 实际: {}", result.pvs[0]);
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

    // 1 秒预算(工具默认配置)在不同局面阶段能到达的深度
    #[test]
    #[ignore]
    fn experiment_depth_bench() {
        let mut mid = startpos_board();
        for mv in ["a0a2", "b2b1", "c0c2", "c3c2", "e0e2", "d2d1", "a2a1", "b3b2", "c2b2", "e2e1"] {
            if game::is_legal(&mid, mv) {
                mid = game::apply_move(&mid, mv);
            }
        }
        let mut late = mid;
        for mv in ["d3d2", "b2b1", "c1c2", "a3a2", "e1e0", "b4b3", "d4d3", "a2a1", "c2c1"] {
            if game::is_legal(&late, mv) {
                late = game::apply_move(&late, mv);
            }
        }
        let mut stage = Camp::Cannon;
        for _ in 0..10 {
            stage = stage.opposite();
        }
        for (name, board) in [("开局", startpos_board()), ("中局", mid), ("残局", late)] {
            let r = analyze(&board, stage, SearchLimits { depth: 30, time_ms: 1000 }, 100).unwrap();
            println!("{name}: 1秒内到达深度 {} 用时{}ms 分数{}", r.depth, r.time_ms, r.score);
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
}
