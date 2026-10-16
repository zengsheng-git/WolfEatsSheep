// 三炮十五兵规则核心
// 棋盘为 5x5 交叉点, board[y][x], y=0 为屏幕最上行, x=0 为最左列
// 棋子: 'C'=炮 'S'=兵 ' '=空; 坐标记法 "c2" = (x=2, y=2), 着法记法 "c2c3"
use serde::Serialize;

pub const SIZE: usize = 5;
pub type Board = [[char; SIZE]; SIZE];

pub const CANNON: char = 'C';
pub const SOLDIER: char = 'S';
pub const EMPTY: char = ' ';

// 兵被吃到该数量以下时炮方获胜(规则: 吃掉最后一枚兵)
pub const SOLDIER_LOSE_COUNT: usize = 0;

// 连续无吃子步数达到该值判和, 防止双方无限空移
pub const DRAW_NO_CAPTURE_PLIES: usize = 60;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize)]
pub enum Camp {
    #[default]
    None,
    Cannon,  // 炮方(先行)
    Soldier, // 兵方
}

impl Camp {
    pub fn to_char(self) -> char {
        match self {
            Camp::Cannon => 'w',
            Camp::Soldier => 'b',
            Camp::None => 'n',
        }
    }

    pub fn from_char(c: char) -> Camp {
        match c {
            'w' => Camp::Cannon,
            'b' => Camp::Soldier,
            _ => Camp::None,
        }
    }

    pub fn opposite(self) -> Camp {
        match self {
            Camp::Cannon => Camp::Soldier,
            Camp::Soldier => Camp::Cannon,
            Camp::None => Camp::None,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Position {
    pub piece: char,
    pub pos: String,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct Changed {
    pub piece: char,
    pub from: String,
    pub to: String,
    pub camp: Camp,
}

#[derive(PartialEq, Eq, Debug)]
pub enum BoardChangeState {
    Move,    // 一子从 from 移到 to(含吃子)
    One,     // 仅单侧变化, 多为动画中间态或误读
    Unknown, // 变化过多, 无法解释
}

const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

pub fn pos_of(x: usize, y: usize) -> String {
    format!("{}{}", (b'a' + x as u8) as char, y)
}

pub fn xy_of(pos: &str) -> Option<(usize, usize)> {
    let bytes = pos.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let x = (bytes[0] as char).to_digit(36).map(|d| d as usize - 10);
    let y = (bytes[1] as char).to_digit(10).map(|d| d as usize);
    match (x, y) {
        (Some(x), Some(y)) if x < SIZE && y < SIZE => Some((x, y)),
        _ => None,
    }
}

// 标准开局: 炮在 a0/c0/e0, 兵填满 2~4 排
pub fn startpos_board() -> Board {
    let mut board = [[EMPTY; SIZE]; SIZE];
    for &x in &[0usize, 2, 4] {
        board[0][x] = CANNON;
    }
    for y in 2..SIZE {
        for x in 0..SIZE {
            board[y][x] = SOLDIER;
        }
    }
    board
}

// 对手视角的开局(炮在下排): 屏幕上炮可能在上方也可能在下方
fn startpos_flipped() -> Board {
    let mut board = [[EMPTY; SIZE]; SIZE];
    for &x in &[0usize, 2, 4] {
        board[SIZE - 1][x] = CANNON;
    }
    for y in 0..3 {
        for x in 0..SIZE {
            board[y][x] = SOLDIER;
        }
    }
    board
}

// 变体开局: 炮在中间三列(微信小程序等平台), 兵方仍占满三行, 其余规则相同
pub fn middle_startpos_board() -> Board {
    let mut board = [[EMPTY; SIZE]; SIZE];
    for x in 1..4 {
        board[0][x] = CANNON;
    }
    for y in 2..SIZE {
        for x in 0..SIZE {
            board[y][x] = SOLDIER;
        }
    }
    board
}

pub fn middle_startpos_flipped() -> Board {
    let mut board = [[EMPTY; SIZE]; SIZE];
    for x in 1..4 {
        board[SIZE - 1][x] = CANNON;
    }
    for y in 0..3 {
        for x in 0..SIZE {
            board[y][x] = SOLDIER;
        }
    }
    board
}

pub fn is_startpos(board: &Board) -> bool {
    *board == startpos_board()
        || *board == startpos_flipped()
        || *board == middle_startpos_board()
        || *board == middle_startpos_flipped()
}

// 某格棋子的合法着法(iccs): 一格平移 + 炮的隔山打吃
pub fn piece_moves(board: &Board, x: usize, y: usize) -> Vec<String> {
    let mut out = Vec::new();
    let piece = board[y][x];
    if piece == EMPTY {
        return out;
    }
    for &(dx, dy) in &DIRS {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if nx < 0 || ny < 0 || nx >= SIZE as i32 || ny >= SIZE as i32 {
            continue;
        }
        let (nx, ny) = (nx as usize, ny as usize);
        if board[ny][nx] == EMPTY {
            out.push(pos_of(x, y) + &pos_of(nx, ny));
        }
    }
    // 炮: 隔一个空格吃兵(民间规则: 越过恰好一个空位, 打掉 2 格外的兵, 中间不能有子;
    // 相邻或隔两个空位都打不到 -- 不是象棋炮的"隔子打")
    if piece == CANNON {
        for &(dx, dy) in &DIRS {
            let tx = x as i32 + dx * 2;
            let ty = y as i32 + dy * 2;
            if tx < 0 || ty < 0 || tx >= SIZE as i32 || ty >= SIZE as i32 {
                continue;
            }
            let (mx, my) = (x as i32 + dx, y as i32 + dy);
            if board[my as usize][mx as usize] == EMPTY && board[ty as usize][tx as usize] == SOLDIER {
                out.push(pos_of(x, y) + &pos_of(tx as usize, ty as usize));
            }
        }
    }
    out
}

pub fn legal_moves(board: &Board, camp: &Camp) -> Vec<String> {
    let want = match camp {
        Camp::Cannon => CANNON,
        Camp::Soldier => SOLDIER,
        Camp::None => return Vec::new(),
    };
    let mut out = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            if board[y][x] == want {
                out.extend(piece_moves(board, x, y));
            }
        }
    }
    out
}

// 落子(吃子即覆盖到点上的兵), 返回新局面
pub fn apply_move(board: &Board, mv: &str) -> Board {
    let mut next = *board;
    let (Some((fx, fy)), Some((tx, ty))) = (xy_of(&mv[..2]), xy_of(&mv[2..])) else {
        return next;
    };
    next[ty][tx] = next[fy][fx];
    next[fy][fx] = EMPTY;
    next
}

// 着法在给定局面上是否合法(由起点棋子类型推导行棋方)
pub fn is_legal(board: &Board, mv: &str) -> bool {
    let Some((fx, fy)) = xy_of(&mv[..2]) else {
        return false;
    };
    piece_moves(board, fx, fy).iter().any(|m| m == mv)
}

pub fn counts(board: &Board) -> (usize, usize) {
    let mut cannons = 0;
    let mut soldiers = 0;
    for row in board.iter() {
        for &cell in row.iter() {
            match cell {
                CANNON => cannons += 1,
                SOLDIER => soldiers += 1,
                _ => {}
            }
        }
    }
    (cannons, soldiers)
}

// 识别结果合理性校验: 炮永远 3 枚(不可被吃), 兵不超过 15
pub fn board_check(board: &Board) -> bool {
    let (cannons, soldiers) = counts(board);
    cannons == 3 && soldiers <= 15
}

// 终局判定: 返回胜方
pub fn winner(board: &Board) -> Option<Camp> {
    let (_, soldiers) = counts(board);
    if soldiers <= SOLDIER_LOSE_COUNT {
        return Some(Camp::Cannon);
    }
    if legal_moves(board, &Camp::Cannon).is_empty() {
        return Some(Camp::Soldier);
    }
    if legal_moves(board, &Camp::Soldier).is_empty() {
        return Some(Camp::Cannon); // 兵全部被堵死同样无力回天
    }
    None
}

// 对比前后两帧局面, 推导发生的一步棋
pub fn board_diff(old: &Board, new: &Board) -> (Changed, BoardChangeState) {
    let mut emptied = Vec::new(); // 变空的格子(候选起点)
    let mut filled = Vec::new();  // 变非空的格子(候选落点)
    for y in 0..SIZE {
        for x in 0..SIZE {
            if old[y][x] != new[y][x] {
                if new[y][x] == EMPTY {
                    emptied.push((x, y));
                } else {
                    filled.push((x, y));
                }
            }
        }
    }
    let changed = |piece: char, from: (usize, usize), to: (usize, usize)| Changed {
        piece,
        from: pos_of(from.0, from.1),
        to: pos_of(to.0, to.1),
        camp: match piece {
            CANNON => Camp::Cannon,
            _ => Camp::Soldier,
        },
    };
    match (emptied.len(), filled.len()) {
        (1, 1) => {
            let piece = new[filled[0].1][filled[0].0];
            (changed(piece, emptied[0], filled[0]), BoardChangeState::Move)
        }
        (1, 0) | (0, 1) => {
            // 单侧变化: 吃子动画中间态或误读
            let (piece, pos) = if emptied.len() == 1 {
                (old[emptied[0].1][emptied[0].0], emptied[0])
            } else {
                (new[filled[0].1][filled[0].0], filled[0])
            };
            (changed(piece, pos, pos), BoardChangeState::One)
        }
        _ => (Changed { piece: EMPTY, from: String::new(), to: String::new(), camp: Camp::None }, BoardChangeState::Unknown),
    }
}

pub fn board_map(board: &Board) -> Vec<Position> {
    let mut out = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            if board[y][x] != EMPTY {
                out.push(Position { piece: board[y][x], pos: pos_of(x, y) });
            }
        }
    }
    out
}

// 中文记谱: 走子 "炮 b1→b2", 吃子 "炮 b1 吃 b4"
pub fn move_text(board: &Board, mv: &str) -> String {
    let (Some((fx, fy)), Some((tx, ty))) = (xy_of(&mv[..2]), xy_of(&mv[2..])) else {
        return mv.to_string();
    };
    let name = if board[fy][fx] == CANNON { "炮" } else { "兵" };
    let from = pos_of(fx, fy);
    let to = pos_of(tx, ty);
    if board[ty][tx] == EMPTY {
        format!("{name} {from}→{to}")
    } else {
        format!("{name} {from} 吃 {to}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_moves() {
        let board = startpos_board();
        // 炮: a0/e0 各 3 种(2 平移 + 1 吃), c0 4 种(3 平移 + 1 吃)
        let cannon_moves = legal_moves(&board, &Camp::Cannon);
        assert_eq!(cannon_moves.len(), 10);
        assert!(cannon_moves.contains(&"a0a2".to_string())); // 开局隔空打: 空位 a1, 吃 a2
        assert!(cannon_moves.contains(&"c0c2".to_string()));
        assert!(cannon_moves.contains(&"e0e2".to_string()));
        // 兵: 只有第 2 排能上移到第 1 排
        let soldier_moves = legal_moves(&board, &Camp::Soldier);
        assert_eq!(soldier_moves.len(), 5);
        assert!(soldier_moves.contains(&"c2c1".to_string()));
    }

    #[test]
    fn test_startpos_recognition() {
        assert!(is_startpos(&startpos_board()));
        assert!(is_startpos(&startpos_flipped()));
        assert!(is_startpos(&middle_startpos_board()));
        assert!(is_startpos(&middle_startpos_flipped()));
        let mut board = startpos_board();
        board[0][0] = EMPTY;
        assert!(!is_startpos(&board));
    }

    #[test]
    fn test_cannon_capture_rules() {
        // 炮隔一个空格吃兵: 中间必须是空位, 距离恰为 2 格
        let mut board = [[EMPTY; SIZE]; SIZE];
        board[0][2] = CANNON; // c0
        board[2][2] = SOLDIER; // c2 兵, c1 为空
        let moves = piece_moves(&board, 2, 0);
        assert!(moves.contains(&"c0c2".to_string()), "隔一空格应能吃兵: {moves:?}");
        // 中间被挡(无射击空间)不能吃
        let mut board2 = board;
        board2[1][2] = SOLDIER; // c1 挡住
        assert!(!piece_moves(&board2, 2, 0).contains(&"c0c2".to_string()), "中间被挡不能吃");
        // 目标是己方炮不能吃
        let mut board3 = [[EMPTY; SIZE]; SIZE];
        board3[0][0] = CANNON;
        board3[2][0] = CANNON;
        assert!(!piece_moves(&board3, 0, 0).contains(&"a0a2".to_string()), "炮不吃炮");
        // 射程只有 2 格
        let mut board4 = [[EMPTY; SIZE]; SIZE];
        board4[0][0] = CANNON;
        board4[3][0] = SOLDIER;
        assert!(!piece_moves(&board4, 0, 0).contains(&"a0a3".to_string()), "隔两个空位打不到");
    }

    #[test]
    fn test_apply_and_legal() {
        let board = startpos_board();
        assert!(is_legal(&board, "a0a2"));
        assert!(!is_legal(&board, "a0a3"), "射程只有 2 格");
        assert!(!is_legal(&board, "a0a4"));
        assert!(!is_legal(&board, "a2a3"), "兵不能吃兵");
        assert!(!is_legal(&board, "a3a2"), "起始位置兵在 a2 不在 a3");
        let next = apply_move(&board, "a0a2");
        assert_eq!(next[2][0], CANNON);
        assert_eq!(next[0][0], EMPTY);
        assert_eq!(counts(&next), (3, 14));
    }

    #[test]
    fn test_winner_soldiers_trap_cannons() {
        // 三门炮均无空邻点且无吃: 兵胜
        let mut board = [[EMPTY; SIZE]; SIZE];
        board[0] = [CANNON, SOLDIER, CANNON, SOLDIER, CANNON];
        board[1] = [SOLDIER, EMPTY, SOLDIER, EMPTY, SOLDIER];
        assert_eq!(winner(&board), Some(Camp::Soldier));
    }

    #[test]
    fn test_winner_soldiers_too_few() {
        // 吃掉最后一枚兵才算炮胜, 剩 2 枚时对局仍继续
        let mut board = [[EMPTY; SIZE]; SIZE];
        board[0][0] = CANNON;
        board[0][2] = CANNON;
        board[0][4] = CANNON;
        board[4][1] = SOLDIER;
        board[4][3] = SOLDIER;
        assert_eq!(winner(&board), None, "剩 2 枚兵对局未结束");
        let mut one = board;
        one[4][1] = EMPTY;
        assert_eq!(winner(&one), None, "剩 1 枚兵对局仍未结束");
        let mut last = one;
        last[4][3] = EMPTY;
        assert_eq!(winner(&last), Some(Camp::Cannon), "最后一枚兵被吃掉才是炮胜");
    }

    #[test]
    fn test_board_diff() {
        let old = startpos_board();
        let new = apply_move(&old, "c0c2");
        let (changed, state) = board_diff(&old, &new);
        assert_eq!(state, BoardChangeState::Move);
        assert_eq!(changed.from, "c0");
        assert_eq!(changed.to, "c2");
        assert_eq!(changed.camp, Camp::Cannon);
        assert_eq!(changed.piece, CANNON);
    }

    #[test]
    fn test_move_text() {
        let board = startpos_board();
        assert_eq!(move_text(&board, "a0a2"), "炮 a0 吃 a2");
        assert_eq!(move_text(&board, "a0a1"), "炮 a0→a1");
        assert_eq!(move_text(&board, "c2c1"), "兵 c2→c1");
    }

    #[test]
    fn test_board_check() {
        assert!(board_check(&startpos_board()));
        let mut bad = startpos_board();
        bad[0][0] = EMPTY; // 炮少了
        assert!(!board_check(&bad));
    }
}
