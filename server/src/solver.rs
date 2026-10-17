// 三炮十五兵全量求解器: 按剩余兵数 k 分层 + 回溯分析(retrograde analysis)
// 方法论参考 0irina/lang-chi-yang-qi(Python) 的分层索引思路, 按本项目规则用 Rust 实现:
// - 状态 = (炮 3 子位置, 兵 k 子位置, 行棋方), k = 剩余兵数 3..=15
//   (k=3 即终局炮胜, k<3 的局面在实战中不可达)
// - 值 = 炮胜 / 兵胜 / 和。和为图语义: 双方都无法强制走向终局(无限互相回避),
//   近似对应运行时的"60 步无吃子判和"(慢速杀棋若需 >60 步静着, 实战可被偷和, 已知偏差)
// - 回溯分析: 终局反向传播。行棋方存在胜着 → 胜; 所有后续皆为对方胜 → 负; 其余 → 和
// - 层间耦合只有吃子(k → k-1): 解第 k 层时读第 k-1 层的已定值(预pass),
//   层内静着构成的反向边在层内广播, 因此各层可自底向上独立求解
//
// 索引: idx = side * (C(25,3) * C(22,k)) + wolf_rank * C(22,k) + sheep_rank
//   side 0=炮行棋 1=兵行棋; 组合序数为 colex(组合数系)
// 表文件 layer_k.bin: 每局面 2 bit(1=炮胜 2=兵胜 3=和), 打包存储



pub const VAL_WOLF: u8 = 1; // 炮胜
pub const VAL_SHEEP: u8 = 2; // 兵胜
pub const VAL_DRAW: u8 = 3; // 和(仅出现在写出的表中; 求解数组中 0=未定即未来和)

const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

const fn build_binom() -> [[u32; 26]; 26] {
    let mut c = [[0u32; 26]; 26];
    let mut n = 0;
    while n < 26 {
        c[n][0] = 1;
        let mut k = 1;
        while k <= n {
            c[n][k] = if k == n { 1 } else { c[n - 1][k - 1] + c[n - 1][k] };
            k += 1;
        }
        n += 1;
    }
    c
}

static BINOM: [[u32; 26]; 26] = build_binom();

pub fn layer_counts(k: usize) -> (u64, u64) {
    // 返回 (每行棋方局面数, C(22,k))
    let c = BINOM[22][k] as u64;
    (2300 * c, c)
}

// colex 组合序数: cells 升序, rank = Σ C(cells[i], i+1)
fn rank_combo(cells: &[u8]) -> u64 {
    let mut r = 0u64;
    for (i, &c) in cells.iter().enumerate() {
        r += BINOM[c as usize][i + 1] as u64;
    }
    r
}

fn unrank_into(mut r: u64, k: usize, out: &mut [u8]) {
    let mut i = k;
    while i >= 1 {
        let mut c = (i - 1) as i32;
        while c <= 25 && (BINOM[c as usize][i] as u64) <= r {
            c += 1;
        }
        c -= 1;
        out[i - 1] = c as u8;
        r -= BINOM[c as usize][i] as u64;
        i -= 1;
    }
}

// 棋盘 → 原始索引。棋盘内部表示: [u8;25], 0空 1炮 2兵, cell = y*5+x
// 升序单遍扫描: 炮/兵都按遇到顺序天然升序, colex 序数可增量累加, 全程无分配
fn rank_board(board: &[u8; 25], k: usize, side: usize) -> u64 {
    let c22k = BINOM[22][k] as u64;
    let mut wolf_rank = 0u64;
    let mut wolf_i = 0usize;
    let mut si = 0usize;
    let mut rem_n = 0u64;
    let mut sheep_rank = 0u64;
    for cell in 0..25u8 {
        if board[cell as usize] == 1 {
            wolf_i += 1;
            wolf_rank += BINOM[cell as usize][wolf_i] as u64;
        } else {
            if board[cell as usize] == 2 {
                si += 1;
                sheep_rank += BINOM[rem_n as usize][si] as u64;
            }
            rem_n += 1;
        }
    }
    side as u64 * 2300 * c22k + wolf_rank * c22k + sheep_rank
}

fn board_from_rank(wolf_rank: u64, sheep_rank: u64, k: usize) -> [u8; 25] {
    let mut board = [0u8; 25];
    let mut wbuf = [0u8; 3];
    unrank_into(wolf_rank, 3, &mut wbuf);
    for &c in wbuf.iter() {
        board[c as usize] = 1;
    }
    // 重建"非炮格升序表"再映射兵局部索引
    let mut rem = [0u8; 22];
    let mut rem_n = 0usize;
    for cell in 0..25u8 {
        if board[cell as usize] == 0 {
            rem[rem_n] = cell;
            rem_n += 1;
        }
    }
    let mut sbuf = [0u8; 15];
    unrank_into(sheep_rank, k, &mut sbuf);
    for &local in sbuf[..k].iter() {
        board[rem[local as usize] as usize] = 2;
    }
    board
}

// 静态局面查询(表已加载时)。返回 Some(值) 或 None(k 超范围)
pub fn table_value(table: &[u8], k: usize, board: &[u8; 25], side: usize) -> u8 {
    let idx = rank_board(board, k, side);
    (table[(idx >> 2) as usize] >> ((idx & 3) * 2)) & 3
}

// 求解单层。prev = 第 k-1 层的打包表(k=4 时传 None, 吃子目标恒为炮胜);
// 取得所有权以便在初始化(唯一用到上一层)结束后立即释放, 降低峰值内存
pub fn solve_layer(k: usize, prev: Option<Vec<u8>>) -> Vec<u8> {
    assert!((3..=15).contains(&k), "k 越界: {k}");
    let (per_side, c22k) = layer_counts(k);
    let n = per_side * 2;
    println!("k={k}: 共 {n} 局面, 初始化(枚举+吃子预pass)...");
    // 每字节: 高 2 位=值(0 未定/1 炮胜/2 兵胜), 低 6 位=未决后续计数
    let mut state: Vec<u8> = vec![0u8; n as usize];
    let mut queue: Vec<u32> = Vec::new();

    // ---- 初始化: 枚举全部局面, 判定终局 / 计数后续 / 预处理吃子边 ----
    let mut board: [u8; 25];
    for a in 0u8..25 {
        for b in (a + 1)..25 {
            for c in (b + 1)..25 {
                let wolves = [a, b, c];
                let mut rem = [0u8; 22];
                let mut rem_n = 0usize;
                for cell in 0..25u8 {
                    if cell != a && cell != b && cell != c {
                        rem[rem_n] = cell;
                        rem_n += 1;
                    }
                }
                let wolf_base = rank_combo(&wolves) * c22k;
                // Gosper 枚举 22 位中恰有 k 个 1 的掩码
                let mut mask: u32 = (1u32 << k) - 1;
                let last: u32 = ((1u32 << k) - 1) << (22 - k);
                loop {
                    board = [0u8; 25];
                    board[a as usize] = 1;
                    board[b as usize] = 1;
                    board[c as usize] = 1;
                    let mut sheep_local = [0u8; 15];
                    let mut si = 0usize;
                    let mut m = mask;
                    while m != 0 {
                        let bit = m.trailing_zeros() as usize;
                        board[rem[bit] as usize] = 2;
                        sheep_local[si] = bit as u8;
                        si += 1;
                        m &= m - 1;
                    }
                    let sheep_rank = rank_combo(&sheep_local[..k]);

                    // 终局判定(与 game::winner 顺序一致: 兵数 → 炮动性 → 兵动性)
                    let wolf_can_move = wolves.iter().any(|&w| piece_has_move(&board, w));
                    let wolf_moves: usize =
                        wolves.iter().map(|&w| piece_move_count(&board, w)).sum();
                    let sheep_can_move = sheep_local[..k]
                        .iter()
                        .any(|&sl| piece_has_move(&board, rem[sl as usize]));
                    let val = if k == 3 {
                        Some(VAL_WOLF)
                    } else if !wolf_can_move {
                        Some(VAL_SHEEP)
                    } else if !sheep_can_move {
                        Some(VAL_WOLF)
                    } else {
                        None
                    };

                    for side in 0..2usize {
                        let idx = (side as u64 * per_side + wolf_base + sheep_rank) as usize;
                        if let Some(v) = val {
                            state[idx] = v << 6;
                            queue.push(idx as u32);
                            continue;
                        }
                        if side == 0 {
                            // 炮行棋: 后续数 = 全部着法; 吃子目标在第 k-1 层, 预先定值/预减
                            let mut rem_cnt = wolf_moves as u8;
                            let mut assigned = 0u8;
                            for &w in wolves.iter() {
                                let (wx, wy) = (w % 5, w / 5);
                                for &(dx, dy) in DIRS.iter() {
                                    let (mx, my) = (wx as i32 + dx, wy as i32 + dy);
                                    let (tx, ty) = (wx as i32 + dx * 2, wy as i32 + dy * 2);
                                    if !(0..5).contains(&mx)
                                        || !(0..5).contains(&my)
                                        || !(0..5).contains(&tx)
                                        || !(0..5).contains(&ty)
                                    {
                                        continue;
                                    }
                                    let (mi, ti) = ((my * 5 + mx) as usize, (ty * 5 + tx) as usize);
                                    if board[mi] != 0 || board[ti] != 2 {
                                        continue;
                                    }
                                    let mut child = board;
                                    child[w as usize] = 0;
                                    child[ti] = 1;
                                    let cval = if k == 4 {
                                        VAL_WOLF // k-1=3 层恒炮胜
                                    } else {
                                        table_value(prev.as_deref().unwrap(), k - 1, &child, 1)
                                    };
                                    match cval {
                                        VAL_WOLF => {
                                            // 炮吃子直接走向炮胜: 立即定值
                                            if assigned == 0 {
                                                assigned = VAL_WOLF;
                                            }
                                        }
                                        VAL_SHEEP => rem_cnt = rem_cnt.saturating_sub(1),
                                        _ => {} // 和: 该后续永不定值, 计数保持(凑不到 0)
                                    }
                                }
                            }
                            if assigned != 0 {
                                state[idx] = assigned << 6;
                                queue.push(idx as u32);
                            } else {
                                state[idx] = rem_cnt & 63;
                            }
                        } else {
                            // 兵行棋: 后续数 = 兵着法数
                            let rem_cnt = sheep_local[..k]
                                .iter()
                                .map(|&sl| piece_move_count(&board, rem[sl as usize]) as u8)
                                .sum::<u8>();
                            state[idx] = rem_cnt & 63;
                        }
                    }

                    if mask == last {
                        break;
                    }
                    mask = gosper_next(mask);
                }
            }
        }
    }

    // ---- 广播: 已定值局面沿静着反向边通知前驱(吃子前驱由上一层预pass覆盖) ----
    // 上一层表只在初始化预pass用到, 此处立即释放以降低峰值内存(k=11/12 时省 ~800MB)
    drop(prev);
    println!("k={k}: 广播求解...");
    let mut drain_count = 0u64;
    while let Some(idx32) = queue.pop() {
        drain_count += 1;
        if drain_count % 200_000_000 == 0 && queue.capacity() > queue.len() * 2 {
            queue.shrink_to_fit(); // 长时间运行的队列容量回收
        }
        let idx = idx32 as u64;
        let side = (idx / per_side) as usize;
        let wolf_rank = (idx % per_side) / c22k;
        let sheep_rank = (idx % per_side) % c22k;
        let val = state[idx as usize] >> 6;
        debug_assert!(val == VAL_WOLF || val == VAL_SHEEP);

        let board = board_from_rank(wolf_rank, sheep_rank, k);
        let mover = 1 - side; // 前驱的行棋方
        let mover_piece = if mover == 0 { 1u8 } else { 2u8 };
        let mut pred = [0u8; 25];
        for t in 0..25usize {
            if board[t] != mover_piece {
                continue;
            }
            let (tx, ty) = (t % 5, t / 5);
            for &(dx, dy) in DIRS.iter() {
                let sx = tx as i32 - dx;
                let sy = ty as i32 - dy;
                if !(0..5).contains(&sx) || !(0..5).contains(&sy) {
                    continue;
                }
                let s = (sy * 5 + sx) as usize;
                if board[s] != 0 {
                    continue;
                }
                // 前驱: 该子在 s, 走到 t
                pred.copy_from_slice(&board);
                pred[s] = mover_piece;
                pred[t] = 0;
                let pi = rank_board(&pred, k, mover) as usize;
                let b = state[pi];
                if b >> 6 != 0 {
                    continue;
                }
                let mover_wins = (val == VAL_WOLF && mover == 0) || (val == VAL_SHEEP && mover == 1);
                if mover_wins {
                    state[pi] = val << 6;
                    queue.push(pi as u32);
                } else {
                    let rem_cnt = (b & 63).saturating_sub(1);
                    if rem_cnt == 0 {
                        // 全部后续皆为该胜方
                        state[pi] = val << 6;
                        queue.push(pi as u32);
                    } else {
                        state[pi] = rem_cnt | (b & 0xC0);
                    }
                }
            }
        }
    }

    // ---- 打包: 未定(0)→和(3) ----
    let mut packed = vec![0u8; ((n + 3) / 4) as usize];
    for i in 0..n as usize {
        let v = match state[i] >> 6 {
            0 => VAL_DRAW,
            v => v,
        };
        packed[i >> 2] |= v << ((i & 3) * 2);
    }
    packed
}

fn gosper_next(v: u32) -> u32 {
    // 标准 Gosper next-permutation: 相同位数的下一个组合
    let t = (v | (v.wrapping_sub(1))).wrapping_add(1);
    t | ((((t & t.wrapping_neg()) / (v & v.wrapping_neg())) >> 1).wrapping_sub(1))
}

fn piece_has_move(board: &[u8; 25], cell: u8) -> bool {
    // 炮的跳吃必须隔空格, 存在跳吃时该空格本身也是平移着法,
    // 因此"有着法"对炮/兵都等价于"存在相邻空点"
    let (x, y) = (cell % 5, cell / 5);
    for &(dx, dy) in DIRS.iter() {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if (0..5).contains(&nx) && (0..5).contains(&ny) && board[(ny * 5 + nx) as usize] == 0 {
            return true;
        }
    }
    false
}

fn piece_move_count(board: &[u8; 25], cell: u8) -> usize {
    let (x, y) = (cell % 5, cell / 5);
    let is_wolf = board[cell as usize] == 1;
    let mut n = 0;
    for &(dx, dy) in DIRS.iter() {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        let mid_empty = (0..5).contains(&nx)
            && (0..5).contains(&ny)
            && board[(ny * 5 + nx) as usize] == 0;
        if !mid_empty {
            continue; // 中间被占: 既不能平移也不能跳吃
        }
        n += 1;
        if is_wolf {
            let tx = x as i32 + dx * 2;
            let ty = y as i32 + dy * 2;
            if (0..5).contains(&tx) && (0..5).contains(&ty) && board[(ty * 5 + tx) as usize] == 2 {
                n += 1;
            }
        }
    }
    n
}

// ---- 表文件读写 ----
pub fn layer_filename(k: usize) -> String {
    format!("layer_{k}.bin")
}

pub fn write_table(dir: &std::path::Path, k: usize, table: &[u8]) -> std::io::Result<()> {
    std::fs::write(dir.join(layer_filename(k)), table)
}

pub fn read_table(dir: &std::path::Path, k: usize) -> Option<Vec<u8>> {
    let (per_side, _) = layer_counts(k);
    let expect = ((per_side * 2 + 3) / 4) as usize;
    let data = std::fs::read(dir.join(layer_filename(k))).ok()?;
    if data.len() != expect {
        return None;
    }
    Some(data)
}

// 统计各值数量(炮胜, 兵胜, 和)
pub fn table_stats(table: &[u8]) -> (u64, u64, u64) {
    let (mut w, mut s, mut d) = (0u64, 0u64, 0u64);
    for &b in table {
        for i in 0..4 {
            match (b >> (i * 2)) & 3 {
                VAL_WOLF => w += 1,
                VAL_SHEEP => s += 1,
                _ => d += 1,
            }
        }
    }
    (w, s, d)
}

#[cfg(test)]
fn board_has_capture(board: &[u8; 25], side: usize) -> bool {
    if side != 0 {
        return false;
    }
    for cell in 0..25usize {
        if board[cell] != 1 {
            continue;
        }
        let (x, y) = (cell % 5, cell / 5);
        for &(dx, dy) in DIRS.iter() {
            let mx = x as i32 + dx;
            let my = y as i32 + dy;
            let tx = x as i32 + dx * 2;
            let ty = y as i32 + dy * 2;
            if (0..5).contains(&mx)
                && (0..5).contains(&my)
                && (0..5).contains(&tx)
                && (0..5).contains(&ty)
                && board[(my * 5 + mx) as usize] == 0
                && board[(ty * 5 + tx) as usize] == 2
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s >> 33
        }
    }

    #[test]
    fn test_combo_rank_roundtrip() {
        let mut rng = lcg(42);
        let mut buf = [0u8; 15];
        for k in [3usize, 5, 9, 13, 15] {
            let total = BINOM[22][k] as u64;
            for _ in 0..200 {
                let r = rng() % total;
                unrank_into(r, k, &mut buf);
                assert_eq!(rank_combo(&buf[..k]), r, "k={k} r={r} cells={:?}", &buf[..k]);
            }
        }
        for _ in 0..200 {
            let r = rng() % 2300;
            unrank_into(r, 3, &mut buf);
            assert_eq!(rank_combo(&buf[..3]), r);
        }
    }

    #[test]
    fn test_gosper_enumerates_all() {
        for k in [3usize, 8, 13] {
            let mut mask: u32 = (1u32 << k) - 1;
            let last: u32 = ((1u32 << k) - 1) << (22 - k);
            let mut count = 1u64;
            while mask != last {
                mask = gosper_next(mask);
                count += 1;
                assert_eq!(mask.count_ones() as usize, k);
            }
            assert_eq!(count, BINOM[22][k] as u64, "k={k}");
        }
    }

    #[test]
    fn test_solve_layer3_all_wolf() {
        let t = solve_layer(3, None);
        let (w, s, d) = table_stats(&t);
        let (per_side, _) = layer_counts(3);
        assert_eq!(w, per_side * 2);
        assert_eq!((s, d), (0, 0), "k=3 层应全部为炮胜");
    }

    // 局部一致性抽查(release 运行): 值与"存在胜着/全败"定义逐点核对
    #[test]
    #[ignore]
    fn experiment_layer4_consistency() {
        let t = solve_layer(4, None);
        let (per_side, c22k) = layer_counts(4);
        let mut checked = 0u64;
        for wolf_rank in 0..2300u64 {
            for sheep_rank in 0..c22k {
                if (wolf_rank + sheep_rank) % 11 != 0 {
                    continue;
                }
                let board = board_from_rank(wolf_rank, sheep_rank, 4);
                for side in 0..2usize {
                    let idx = side as u64 * per_side + wolf_rank * c22k + sheep_rank;
                    let v = (t[(idx >> 2) as usize] >> ((idx & 3) * 2)) & 3;
                    // 静着后继值集合
                    let mut quiet_vals = Vec::new();
                    let piece = if side == 0 { 1u8 } else { 2u8 };
                    for cell in 0..25usize {
                        if board[cell] != piece {
                            continue;
                        }
                        let (x, y) = (cell % 5, cell / 5);
                        for &(dx, dy) in DIRS.iter() {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if (0..5).contains(&nx) && (0..5).contains(&ny) {
                                let ni = (ny * 5 + nx) as usize;
                                if board[ni] == 0 {
                                    let mut child = board;
                                    child[ni] = piece;
                                    child[cell] = 0;
                                    quiet_vals.push(table_value(&t, 4, &child, 1 - side));
                                }
                            }
                        }
                    }
                    if v == VAL_WOLF {
                        // k=4 的吃子后继落在 k=3 层, 恒为炮胜
                        let has_capture = side == 0 && board_has_capture(&board, side);
                        if side == 0 {
                            assert!(
                                has_capture || quiet_vals.contains(&VAL_WOLF),
                                "炮方应存在炮胜后续"
                            );
                        } else {
                            assert!(
                                quiet_vals.iter().all(|&c| c == VAL_WOLF),
                                "兵方炮败局面所有后续应为炮胜"
                            );
                        }
                    } else if v == VAL_SHEEP {
                        if side == 1 {
                            assert!(quiet_vals.contains(&VAL_SHEEP), "兵方应存在兵胜后续");
                        } else {
                            assert!(
                                !board_has_capture(&board, side)
                                    && quiet_vals.iter().all(|&c| c == VAL_SHEEP),
                                "炮方兵败局面不应有吃子且所有后续应为兵胜"
                            );
                        }
                    } else {
                        // 和: 后续全为 和 或 行棋方失利方胜, 且至少一个和; 炮方不应有吃子(否则早被定值为炮胜)
                        let bad = if side == 0 { VAL_SHEEP } else { VAL_WOLF };
                        assert!(
                            quiet_vals.iter().all(|&c| c == VAL_DRAW || c == bad),
                            "和棋局面不应存在更优后续"
                        );
                        assert!(
                            quiet_vals.contains(&VAL_DRAW),
                            "和棋局面应至少存在一个和后续"
                        );
                        if side == 0 {
                            assert!(!board_has_capture(&board, side), "和棋局面炮方不应有吃子");
                        }
                    }
                    checked += 1;
                }
            }
        }
        let (w, s, d) = table_stats(&t);
        println!("k=4 层: 炮胜{w} 兵胜{s} 和{d} 抽查{checked}个局面全部一致");
    }
}
