// 三炮十五兵棋盘识别: 经典 CV 管线, 替代原项目的 YOLO ONNX 模型
// 思路: 棋子是棋盘上唯一的"圆形色块" -> 连通域提取 -> 拟合 5x5 网格
//       -> 自举两方颜色质心 -> 后续帧按网格归属 + 最近色心分类
// 网格拟合有两条路: 开局阵型指纹(首帧恰为开局时的可靠路径)与任意局面拟合
// (等距格点 RANSAC, 对局进行中接入监听时使用, 见 fit_arbitrary)
// 网格线若颜色饱和会被一并掩码, 且线条穿过棋子中心会把棋子连通成片;
// 对此提供腐蚀通道(5x5 腐蚀切断细线、保留实心棋子), 主通道指纹失败时自动启用
use xcap::image::{ImageBuffer, Rgba};

use crate::game::{self, Board, CANNON, EMPTY, SOLDIER};

// 饱和度阈值: 高于它视为"彩色"(棋子), 木质等低饱和背景被排除
const SAT_MIN: u8 = 90;
// 亮度阈值: 低于它视为"深色"(黑字/黑子)
const VAL_MAX: u8 = 80;

#[derive(Clone, Debug)]
pub struct GridModel {
    /// 5 个交叉点的 x 坐标(相对裁剪后图像, 原始像素)
    pub xs: Vec<f32>,
    /// 5 个交叉点的 y 坐标
    pub ys: Vec<f32>,
    /// 平均格距(像素)
    pub cell: f32,
    /// 两方颜色质心(RGB)
    pub cannon_color: [f32; 3],
    pub soldier_color: [f32; 3],
    /// 屏幕上炮方是否在上方
    pub cannons_top: bool,
    /// 识别使用的通道: true=腐蚀通道(网格线被掩码且穿过棋子时启用)
    pub eroded: bool,
    /// true=字形通道(黑字兵/红字炮, 橙红底平台如微信小程序)
    pub glyph: bool,
}

impl GridModel {
    /// 网格整体平移 k 格(正 = 向右/向下)。任意局面启动时端部空行/列的归属可能
    /// 判错(棋盘相对识别整体错位一格), "纠"按钮发现标注局面与平移后的识别一致
    /// 时调用它对齐; 格点等距, 均匀平移即等价于重排窗口
    pub fn shifted(&self, dx: i32, dy: i32) -> GridModel {
        let mut m = self.clone();
        let sx = dx as f32 * self.cell;
        let sy = dy as f32 * self.cell;
        for x in &mut m.xs {
            *x += sx;
        }
        for y in &mut m.ys {
            *y += sy;
        }
        m
    }
}


pub struct FirstDetect {
    pub board: Board,
    pub model: GridModel,
    /// 棋盘裁剪框(原始图像坐标 x, y, w, h)
    pub crop: (u32, u32, u32, u32),
}

struct Blob {
    cx: f32,
    cy: f32,
    area: usize,
    rgb: [f64; 3],
    colored: usize, // 高饱和像素数(0 = 纯深色块, 如炮位空槽等装饰)
}

// 提取圆形色块: 颜色阈值分割 + (可选)形态学处理 + 连通域 + 形状/填充率/面积过滤
// erode=0 为主通道; erode=r 时像素需在 (2r+1)^2 邻域内全部被掩码才保留(切断细线)
// 返回 (色块列表, 采样步长); 色块坐标为采样网格坐标(乘 step 即原始像素)
fn extract_blobs(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    max_width: usize,
    erode: usize,
    glyph: bool,
) -> (Vec<Blob>, usize) {
    let (w, h) = img.dimensions();
    let step = (((w as usize) + max_width - 1) / max_width).max(1);
    let sw = ((w as usize) + step - 1) / step;
    let sh = ((h as usize) + step - 1) / step;

    let is_masked = |sx: usize, sy: usize| -> bool {
        let px = (sx * step).min(w as usize - 1) as u32;
        let py = (sy * step).min(h as usize - 1) as u32;
        let p = img.get_pixel(px, py);
        let maxc = p[0].max(p[1]).max(p[2]);
        let minc = p[0].min(p[1]).min(p[2]);
        let sat = maxc - minc;
        if glyph {
            // 字形通道(橙红底平台如微信小程序): 棋子由字形标识 --
            // 黑字(兵, 含笔芯暗阶; 盘面暗影带 sat>=60 被排除) + 红字(炮, 强红且绿蓝双低,
            // 阈值卡在字形暗红与盘环暗橙之间)
            let black_ink = maxc < 170 && sat < 60;
            let red_glyph = p[0] > 120 && p[1] < 90 && p[2] < 90 && p[0] > p[1] + 90;
            black_ink || red_glyph
        } else {
            // 彩色通道: 彩色棋子(实心彩色盘) / 带彩深色 / 近白字形笔画(避免白字切碎盘面);
            // 深色须带彩度(排除棋盘投影等无彩阴影); 近白窗口排除黄调木底(蓝通道明显低于红)
            sat > SAT_MIN || (maxc < VAL_MAX && sat > 15) || (maxc > 232 && sat < 30 && p[2] >= p[0].saturating_sub(10))
        }
    };

    // 先物化掩码数组(形态学处理与洪泛共用)
    let mut mask = vec![false; sw * sh];
    for sy in 0..sh {
        for sx in 0..sw {
            mask[sy * sw + sx] = is_masked(sx, sy);
        }
    }
    if erode == 0 {
        // 主通道: 先抹除大型空心环/框(棋盘投影阴影等, 膨胀后易与边缘棋子粘连)
        remove_hollow(&mut mask, sw, sh);
    } else {
        // 腐蚀通道: 只腐蚀, 切断穿过棋子的饱和细网格线
        mask = erode_mask(&mask, sw, sh, erode);
    }

    // 字形笔画会在棋子盘面留下缝隙: 洪泛时允许沿直线跨越不超过 GAP_JUMP 的
    // 非掩码窄缝连接两侧盘面(相邻棋子间距远大于该值, 不会被误连)
    const GAP_JUMP: i32 = 6;
    let pixel = |sx: usize, sy: usize| -> [f64; 3] {
        let px = (sx * step).min(w as usize - 1) as u32;
        let py = (sy * step).min(h as usize - 1) as u32;
        let p = img.get_pixel(px, py);
        [p[0] as f64, p[1] as f64, p[2] as f64]
    };

    let mut visited = vec![false; sw * sh];
    let mut blobs = Vec::new();
    for sy in 0..sh {
        for sx in 0..sw {
            if visited[sy * sw + sx] || !mask[sy * sw + sx] {
                continue;
            }
            // 迭代式洪泛填充(四连通 + 跨窄缝)
            let mut stack = vec![(sx, sy)];
            visited[sy * sw + sx] = true;
            let mut area = 0usize;
            let mut sum_x = 0f64;
            let mut sum_y = 0f64;
            let mut sum_rgb = [0f64; 3];
            // 颜色统计只用彩色像素, 避免字形笔画把均色拉白
            let mut color_area = 0usize;
            let mut color_rgb = [0f64; 3];
            let mut min_x = sx;
            let mut max_x = sx;
            let mut min_y = sy;
            let mut max_y = sy;
            while let Some((x, y)) = stack.pop() {
                area += 1;
                sum_x += x as f64;
                sum_y += y as f64;
                let rgb = pixel(x, y);
                sum_rgb = add3(sum_rgb, rgb);
                if (rgb[0].max(rgb[1]).max(rgb[2]) - rgb[0].min(rgb[1]).min(rgb[2])) > SAT_MIN as f64 {
                    color_area += 1;
                    color_rgb = add3(color_rgb, rgb);
                }
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    for j in 1..=GAP_JUMP {
                        let nx = x as i32 + dx * j;
                        let ny = y as i32 + dy * j;
                        if nx < 0 || ny < 0 || nx >= sw as i32 || ny >= sh as i32 {
                            break;
                        }
                        let (nx, ny) = (nx as usize, ny as usize);
                        if mask[ny * sw + nx] {
                            if !visited[ny * sw + nx] {
                                visited[ny * sw + nx] = true;
                                stack.push((nx, ny));
                            }
                            break; // 该方向遇到掩码(直接相邻或跨缝)即停
                        }
                        // 非掩码缝隙: 继续向远处跳
                    }
                }
            }
            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            // 形状过滤: 棋子近圆形, 排除细长的线条/边框
            let long_side = bw.max(bh);
            let short_side = bw.min(bh).max(1);
            if long_side > short_side * 13 / 10 + 2 {
                #[cfg(test)]
                if std::env::var("SANPAO_DEBUG").is_ok() {
                    println!("  [aspect剔除] at({sx},{sy}) {bw}x{bh} area={area}");
                }
                continue;
            }
            // 填充率过滤: 实心圆约 0.79, 字形约 0.3+, 空心网格/方框远低于此
            let fill = area as f32 / (bw * bh) as f32;
            if fill < 0.2 || fill > 0.95 {
                #[cfg(test)]
                if std::env::var("SANPAO_DEBUG").is_ok() {
                    println!("  [fill剔除] at({sx},{sy}) fill={fill:.2} area={area}");
                }
                continue;
            }
            if area < 6 {
                #[cfg(test)]
                if std::env::var("SANPAO_DEBUG").is_ok() {
                    println!("  [面积剔除] at({sx},{sy}) area={area}");
                }
                continue;
            }
            blobs.push(Blob {
                cx: (sum_x / area as f64) as f32,
                cy: (sum_y / area as f64) as f32,
                area,
                rgb: if color_area > 0 { scale3(color_rgb, 1.0 / color_area as f64) } else { scale3(sum_rgb, 1.0 / area as f64) },
                colored: color_area,
            });
        }
    }

    // 面积中位数过滤(迭代至稳定): 保留与典型棋子尺寸同量级的色块, 逐轮剔除离群的大背景块/小噪声
    for _ in 0..3 {
        if blobs.len() < 3 {
            break;
        }
        let mut areas: Vec<usize> = blobs.iter().map(|b| b.area).collect();
        areas.sort_unstable();
        let med = areas[areas.len() / 2] as f32;
        let before = blobs.len();
        let kept: Vec<Blob> = blobs.drain(..).filter(|b| {
            // 上界放宽到 6 倍: 黑字浅盘平台上, 字形棋子(小)与实心炮盘(大)两类尺寸共存
            let ok = (b.area as f32) > med * 0.25 && (b.area as f32) < med * 6.0;
            #[cfg(test)]
            if !ok && std::env::var("SANPAO_DEBUG").is_ok() {
                println!("  [中位数剔除] med={med:.0} area={} at({:.0},{:.0})", b.area, b.cx, b.cy);
            }
            ok
        }).collect();
        blobs = kept;
        if blobs.len() == before {
            break;
        }
    }
    (blobs, step)
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// 腐蚀: (2r+1)^2 邻域全部为掩码才保留; 触边视为不掩码
fn erode_mask(mask: &[bool], sw: usize, sh: usize, r: usize) -> Vec<bool> {
    (0..sh)
        .flat_map(|sy| (0..sw).map(move |sx| (sx, sy)))
        .map(|(sx, sy)| {
            if sx < r || sy < r || sx + r >= sw || sy + r >= sh {
                return false;
            }
            (sy - r..=sy + r).all(|ny| (sx - r..=sx + r).all(|nx| mask[ny * sw + nx]))
        })
        .collect()
}

// 抹除大型空心环/框结构: 连通域的包围盒填充率过低时整体移除
// (棋盘 div 的投影阴影是环绕棋盘的空心框, 若不先移除, 膨胀后会与贴边棋子粘连)
fn remove_hollow(mask: &mut Vec<bool>, sw: usize, sh: usize) {
    let mut visited = vec![false; sw * sh];
    let mut remove = Vec::new();
    for sy in 0..sh {
        for sx in 0..sw {
            if visited[sy * sw + sx] || !mask[sy * sw + sx] {
                continue;
            }
            let mut stack = vec![(sx, sy)];
            visited[sy * sw + sx] = true;
            let mut cells = Vec::new();
            let mut min_x = sx;
            let mut max_x = sx;
            let mut min_y = sy;
            let mut max_y = sy;
            while let Some((x, y)) = stack.pop() {
                cells.push((x, y));
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= sw as i32 || ny >= sh as i32 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if !visited[ny * sw + nx] && mask[ny * sw + nx] {
                        visited[ny * sw + nx] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            let fill = cells.len() as f32 / (bw * bh) as f32;
            // 小型空心结构(字形洞/黑字棋子的笔画框等)保留, 只清大块
            if fill < 0.35 && cells.len() > 800 {
                #[cfg(test)]
                if std::env::var("SANPAO_DEBUG").is_ok() {
                    println!("  [remove_hollow] at({min_x},{min_y}) {bw}x{bh} cells={}", cells.len());
                }
                remove.extend(cells);
            }
        }
    }
    for (x, y) in remove {
        mask[y * sw + x] = false;
    }
}

fn scale3(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn color_dist(a: &[f64; 3], b: &[f32; 3]) -> f32 {
    let dr = a[0] - b[0] as f64;
    let dg = a[1] - b[1] as f64;
    let db = a[2] - b[2] as f64;
    (dr * dr + dg * dg + db * db) as f32
}

// 一维 k 均值, 多种初始化重启取簇内误差最小者:
// 单一初始化(如 min-max 等距)在存在空行时会把中心落在空行上导致行合并
fn kmeans1d(values: &[f32], k: usize, iters: usize) -> Vec<(f32, usize)> {
    let n = values.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // Lloyd 迭代: 分配 -> 更新, 空簇重置为离旧中心最远的数据点; 返回 (中心, 分配, 簇内平方误差)
    let lloyd = |mut centers: Vec<f32>| -> (Vec<f32>, Vec<usize>, f64) {
        let mut assign = vec![0usize; n];
        let nearest = |centers: &[f32], v: f32| -> (usize, f32) {
            centers
                .iter()
                .enumerate()
                .map(|(c, ctr)| (c, (ctr - v).abs()))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
        };
        for _ in 0..iters {
            let mut changed = false;
            for (i, v) in values.iter().enumerate() {
                let (best, _) = nearest(&centers, *v);
                if assign[i] != best {
                    assign[i] = best;
                    changed = true;
                }
            }
            for c in 0..k {
                let mut sum = 0f32;
                let mut cnt = 0usize;
                for (i, v) in values.iter().enumerate() {
                    if assign[i] == c {
                        sum += v;
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    // 空簇: 重置为离旧中心最远的数据点, 避免簇永久死亡
                    if let Some((_, far)) = values
                        .iter()
                        .map(|v| ((centers[c] - v).abs(), v))
                        .max_by(|a, b| a.0.total_cmp(&b.0))
                    {
                        centers[c] = *far;
                        changed = true;
                    }
                } else {
                    centers[c] = sum / cnt as f32;
                }
            }
            if !changed {
                break;
            }
        }
        let mut sse = 0f64;
        for (i, v) in values.iter().enumerate() {
            let (best, d) = nearest(&centers, *v);
            assign[i] = best;
            sse += (d * d) as f64;
        }
        (centers, assign, sse)
    };

    // 候选初始化: 等距 + 分位数 + 确定性随机采样
    let mut cands: Vec<Vec<f32>> = Vec::new();
    let min = sorted[0];
    let max = sorted[n - 1];
    cands.push(
        (0..k)
            .map(|i| min + (max - min) * i as f32 / (k as f32 - 1.0).max(1.0))
            .collect(),
    );
    cands.push((0..k).map(|i| sorted[(((i + 1) * n) / k).saturating_sub(1).min(n - 1)]).collect());
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..12 {
        let sample: Vec<f32> = (0..k)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                values[((rng >> 33) as usize) % n]
            })
            .collect();
        cands.push(sample);
    }

    let mut best: Option<(Vec<f32>, Vec<usize>, f64)> = None;
    for cand in cands {
        let result = lloyd(cand);
        if best.as_ref().map(|b| result.2 < b.2).unwrap_or(true) {
            best = Some(result);
        }
    }
    let (centers, assign, _) = best.unwrap();
    let mut out: Vec<(f32, usize)> = centers
        .iter()
        .enumerate()
        .map(|(c, center)| (*center, assign.iter().filter(|a| **a == c).count()))
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

fn flipped_startpos() -> Board {
    let mut board = [[EMPTY; game::SIZE]; game::SIZE];
    for &x in &[0usize, 2, 4] {
        board[game::SIZE - 1][x] = CANNON;
    }
    for y in 0..3 {
        for x in 0..game::SIZE {
            board[y][x] = SOLDIER;
        }
    }
    board
}

fn nearest_index(values: &[f32], v: f32) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for (i, c) in values.iter().enumerate() {
        let d = (c - v).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

// 首帧识别: 依次尝试多个通道, 任一通道拟合出开局指纹或任意局面网格即成功
// 通道 = (是否字形掩码, 腐蚀半径):
//   实心彩色棋子平台走彩色通道; 橙红底+黑红字平台(微信小程序)走字形通道;
//   腐蚀变体切断被掩码的网格线, 避免线条把棋子连成片
pub fn detect_first(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, hint: Option<&ColorHint>) -> Result<FirstDetect, String> {
    let (iw, ih) = img.dimensions();
    let channels: [(bool, usize); 4] = [(false, 0), (false, 2), (true, 0), (true, 2)];
    // 全部失败时返回"走得最远"(色块最多)的通道的错误, 对用户信息量最大
    let mut best_err: Option<(String, usize)> = None;
    for &(glyph, erode) in &channels {
        let (blobs, step) = extract_blobs(img, 1024, erode, glyph);
        #[cfg(test)]
        if std::env::var("SANPAO_DEBUG").is_ok() {
            println!("== channel glyph={glyph} erode={erode}: {} blobs", blobs.len());
        }
        // 开局指纹优先(开局局面的既有可靠路径), 失败再按任意局面拟合,
        // 让监听可以从对局进行中的任意局面启动
        if let Ok(det) = fit_startpos(&blobs, step, iw, ih, glyph, erode != 0) {
            return Ok(det);
        }
        match fit_arbitrary(img, &blobs, step, glyph, erode != 0, hint) {
            Ok(det) => return Ok(det),
            Err(e) => {
                #[cfg(test)]
                if std::env::var("SANPAO_DEBUG").is_ok() {
                    println!("  [fit_arbitrary] glyph={glyph} erode={erode} 失败: {e}");
                }
                if best_err.as_ref().map(|(_, n)| blobs.len() > *n).unwrap_or(true) {
                    best_err = Some((e, blobs.len()));
                }
            }
        }
    }
    Err(best_err.map(|(e, _)| e).unwrap_or_else(|| "未识别到棋盘".to_string()))
}

// 由色块集合拟合开局指纹: 行列聚类 + 几何校验 + 颜色自举
// 真实窗口常混有头像/文字/按钮等噪声色块: 逐级加大聚类簇数,
// 在簇中寻找开局指纹(3 个整行 + 端点炮行 / 5 个整列), 其余簇视为噪声忽略
fn fit_startpos(blobs: &[Blob], step: usize, iw: u32, ih: u32, glyph: bool, eroded: bool) -> Result<FirstDetect, String> {
    if blobs.len() < 18 {
        return Err(format!("识别到 {} 个棋子色块, 需要至少 18 个(标准开局), 请确保目标窗口完整显示棋盘", blobs.len()));
    }
    #[cfg(test)]
    if std::env::var("SANPAO_DEBUG").is_ok() {
        for b in blobs {
            println!("  [blob] ({:.0},{:.0}) area={} rgb=({:.0},{:.0},{:.0})", b.cx, b.cy, b.area, b.rgb[0], b.rgb[1], b.rgb[2]);
        }
    }

    // 行指纹: 连续 3 个整行(各 5 枚) + 相距约 2 格的炮行(3 枚)位于整行段一端
    let ys: Vec<f32> = blobs.iter().map(|b| b.cy).collect();
    let mut row_hit = None;
    let mut last_rows: Vec<usize> = Vec::new();
    for k in 4..=8usize {
        let rows = kmeans1d(&ys, k, 40);
        last_rows = rows.iter().map(|r| r.1).collect();
        if let Some(p) = find_row_pattern(&rows) {
            row_hit = Some(p);
            break;
        }
    }
    let Some((soldier_rows, cannon_row_y, cannons_top)) = row_hit else {
        return Err(format!("棋子行布局与标准开局不符({last_rows:?}), 请从开局局面开始监听"));
    };
    let spacing_y = (soldier_rows[1] - soldier_rows[0] + soldier_rows[2] - soldier_rows[1]) / 2.0;

    // 行归属: 距行中心半格以内视为该行棋子, 其余为行间噪声; 装饰块(空槽等)一并排除
    let deco = make_deco_filter(glyph, blobs);
    let in_row = |b: &Blob, row_y: f32| (b.cy - row_y).abs() < spacing_y * 0.5;
    let soldier_blobs: Vec<&Blob> = blobs
        .iter()
        .filter(|b| !deco(b) && soldier_rows.iter().any(|&ry| in_row(b, ry)))
        .collect();
    if soldier_blobs.len() != 15 {
        return Err(format!("兵方棋子数异常({}), 请从开局局面开始监听", soldier_blobs.len()));
    }
    let cannon_blobs: Vec<&Blob> = blobs.iter().filter(|b| !deco(b) && in_row(b, cannon_row_y)).collect();
    if cannon_blobs.len() != 3 {
        return Err("炮的数量异常, 请从开局局面开始监听".to_string());
    }

    // 列指纹: 5 个各 3 枚的等距列
    let xs: Vec<f32> = soldier_blobs.iter().map(|b| b.cx).collect();
    let mut col_hit = None;
    let mut last_cols: Vec<usize> = Vec::new();
    for k in 5..=8usize {
        let cols = kmeans1d(&xs, k, 40);
        last_cols = cols.iter().map(|c| c.1).collect();
        if let Some(p) = find_col_pattern(&cols) {
            col_hit = Some(p);
            break;
        }
    }
    let Some(col_centers) = col_hit else {
        return Err(format!("棋盘列布局异常({last_cols:?}), 识别失败"));
    };
    let dxs: Vec<f32> = (1..5).map(|i| col_centers[i] - col_centers[i - 1]).collect();
    let spacing_x = (dxs[0] + dxs[1] + dxs[2] + dxs[3]) / 4.0;
    if (spacing_x - spacing_y).abs() > spacing_x.max(spacing_y) * 0.35 {
        return Err("棋盘格子不是正方形, 识别失败".to_string());
    }

    // 炮的列组合必须是 {0,2,4}(炮分立, 标准开局)或 {1,2,3}(炮居中, 小程序变体)
    let mut cannon_cols = [false; 5];
    for b in cannon_blobs.iter() {
        let col = nearest_index(&col_centers, b.cx);
        if (b.cx - col_centers[col]).abs() >= spacing_x * 0.45 {
            return Err("炮不在标准列上, 请从开局局面开始监听".to_string());
        }
        cannon_cols[col] = true;
    }
    if cannon_cols != [true, false, true, false, true] && cannon_cols != [false, true, true, true, false] {
        return Err("炮列布局与已知开局不符, 请从开局局面开始监听".to_string());
    }

    // 构造棋盘(采样坐标 -> 逻辑格)
    let mut board = [[EMPTY; game::SIZE]; game::SIZE];
    let cannon_y = if cannons_top { 0 } else { game::SIZE - 1 };
    for b in cannon_blobs.iter() {
        board[cannon_y][nearest_index(&col_centers, b.cx)] = CANNON;
    }
    for b in soldier_blobs.iter() {
        let col = nearest_index(&col_centers, b.cx);
        let soldier_row = nearest_index(&soldier_rows, b.cy);
        let y = if cannons_top { soldier_row + 2 } else { soldier_row };
        board[y][col] = SOLDIER;
    }
    if board != game::startpos_board() && board != flipped_startpos() && board != game::middle_startpos_board() && board != game::middle_startpos_flipped() {
        return Err("开局布局校验失败, 请从标准开局局面开始监听".to_string());
    }

    // 完整网格: 在炮行与兵行之间补插空行
    let mut lattice_ys: Vec<f32> = Vec::new();
    if cannons_top {
        lattice_ys.push(cannon_row_y);
        lattice_ys.push(soldier_rows[0] - spacing_y);
        lattice_ys.extend_from_slice(&soldier_rows);
    } else {
        lattice_ys.extend_from_slice(&soldier_rows);
        lattice_ys.push(soldier_rows[2] + spacing_y);
        lattice_ys.push(cannon_row_y);
    }

    let mean_color = |bs: &[&Blob]| -> [f32; 3] {
        let mut acc = [0f64; 3];
        for b in bs {
            acc = add3(acc, b.rgb);
        }
        scale3(acc, 1.0 / bs.len().max(1) as f64).map(|v| v as f32)
    };

    let cell = (spacing_x + spacing_y) / 2.0;
    // 裁剪框: 网格外扩 0.8 格, 转换到原始像素坐标
    let scale = step as f32;
    let x0 = (((col_centers[0] - spacing_x * 0.8).max(0.0)) * scale) as u32;
    let y0 = (((lattice_ys[0] - spacing_y * 0.8).max(0.0)) * scale) as u32;
    let x1 = (((col_centers[4] + spacing_x * 0.8) * scale) as u32).min(iw);
    let y1 = (((lattice_ys[4] + spacing_y * 0.8) * scale) as u32).min(ih);
    let crop = (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0));

    let model = GridModel {
        xs: col_centers.iter().map(|x| x * scale - x0 as f32).collect(),
        ys: lattice_ys.iter().map(|y| y * scale - y0 as f32).collect(),
        cell: cell * scale,
        cannon_color: mean_color(&cannon_blobs),
        soldier_color: mean_color(&soldier_blobs),
        cannons_top,
        eroded,
        glyph,
    };
    Ok(FirstDetect { board, model, crop })
}

// 在行簇中寻找开局指纹: 选出 3 个 5 枚整行(等距连续), 再找一个 3 枚簇
// 位于整行段一端约 2 格处充当炮行; 不参与的簇(界面噪声)被忽略
fn find_row_pattern(rows: &[(f32, usize)]) -> Option<(Vec<f32>, f32, bool)> {
    let full: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].1 == 5).collect();
    if full.len() < 3 {
        return None;
    }
    for trio in combos(&full, 3) {
        let ys: Vec<f32> = trio.iter().map(|&i| rows[i].0).collect();
        let d1 = ys[1] - ys[0];
        let d2 = ys[2] - ys[1];
        let spacing = (d1 + d2) / 2.0;
        if spacing <= 0.0 || (d1 - d2).abs() > spacing * 0.35 {
            continue;
        }
        for ci in (0..rows.len()).filter(|&i| rows[i].1 == 3) {
            let cy = rows[ci].0;
            let cannons_top = cy < ys[0];
            let gap = if cannons_top { ys[0] - cy } else { cy - ys[2] };
            if (gap - spacing * 2.0).abs() <= spacing * 0.4 {
                return Some((ys, cy, cannons_top));
            }
        }
    }
    None
}

// 在列簇中寻找 5 个各 3 枚的等距列
fn find_col_pattern(cols: &[(f32, usize)]) -> Option<Vec<f32>> {
    let cand: Vec<usize> = (0..cols.len()).filter(|&i| cols[i].1 == 3).collect();
    if cand.len() < 5 {
        return None;
    }
    for quint in combos(&cand, 5) {
        let xs: Vec<f32> = quint.iter().map(|&i| cols[i].0).collect();
        let dxs: Vec<f32> = (1..5).map(|i| xs[i] - xs[i - 1]).collect();
        let spacing = (dxs[0] + dxs[1] + dxs[2] + dxs[3]) / 4.0;
        if spacing > 0.0 && dxs.iter().all(|d| (d - spacing).abs() <= spacing * 0.3) {
            return Some(xs);
        }
    }
    None
}

// 升序索引的全部 k 组合(字典序)
fn combos(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    fn rec(items: &[usize], k: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..items.len() {
            cur.push(items[i]);
            rec(items, k, i + 1, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    if items.len() >= k {
        rec(items, k, 0, &mut Vec::new(), &mut out);
    }
    out
}


// ---- 任意局面自举(不依赖开局指纹) ----

// 一维等距格点拟合: 相位按 s/40 步长细扫(而非锚定在色块上——腐蚀通道下
// 兵/炮字形质心向相反方向偏离格点, 真实格点可能在任何色块之间),
// 取"5 格窗口内点数"最大的 (间距, 相位); 同分保留更大间距。
// 吸附容差 0.22 格: 需同时罩住兵(~-0.17 格)与炮(~+0.13 格)的质心偏移
const SNAP_FRAC: f32 = 0.22;

// 一个候选窗口: 窗口内点数、校准后的间距与 5 个格点坐标(升序)
struct WindowFit {
    score: usize,
    spacing: f32,
    positions: Vec<f32>,
}

// 一维等距格点拟合结果: 接近最高分的全部窗口候选(按得分降序、位置靠上优先)。
// 棋盘外的装饰行(金色标记/底部按钮)或端部整行空出都可能让最高分窗口不是
// 真棋盘, 真伪由上层结合棋子分类校验(炮恒 3)选定
struct AxisFit {
    windows: Vec<WindowFit>,
    spacing: f32,
    /// 得分最高窗口中有色块吸附的不同格点数
    distinct: usize,
}

fn best_lattice(coords: &[f32], spacings: &[f32]) -> Option<AxisFit> {
    let n = coords.len();
    if n < 2 || spacings.is_empty() {
        return None;
    }
    let snap = |s: f32, phase: f32| -> Vec<Option<i32>> {
        coords
            .iter()
            .map(|&c| {
                let k = ((c - phase) / s).round() as i32;
                if (c - (phase + k as f32 * s)).abs() <= s * SNAP_FRAC {
                    Some(k)
                } else {
                    None
                }
            })
            .collect()
    };
    // 5 格窗口 [lo, lo+4] 内的色块数; 窗口外的"内点"按噪声不计分
    let windows_of = |ks: &[Option<i32>]| -> Vec<(usize, i32)> {
        let flat: Vec<i32> = ks.iter().flatten().copied().collect();
        if flat.is_empty() {
            return Vec::new();
        }
        let kmin = *flat.iter().min().unwrap();
        let kmax = *flat.iter().max().unwrap();
        (kmin..=kmax).map(|lo| (flat.iter().filter(|&&k| k >= lo && k <= lo + 4).count(), lo)).collect()
    };

    let mut sorted = spacings.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    // 每个间距的局部最优 (得分, 相位): 按 s/40 步长细扫全周期
    let mut per_s: Vec<(usize, f32, f32)> = Vec::new();
    #[cfg(test)]
    let mut all_scores: Vec<(usize, f32, f32)> = Vec::new();
    for &s in &sorted {
        let mut local_best: Option<(usize, f32)> = None;
        for si in 0..40 {
            let phase = s * si as f32 / 40.0;
            let Some(score) = windows_of(&snap(s, phase)).iter().map(|w| w.0).max() else {
                continue;
            };
            if local_best.as_ref().map(|b| score > b.0).unwrap_or(true) {
                local_best = Some((score, phase));
            }
        }
        if let Some((score, phase)) = local_best {
            #[cfg(test)]
            all_scores.push((score, s, phase));
            per_s.push((score, s, phase));
        }
    }
    #[cfg(test)]
    if std::env::var("SANPAO_DEBUG").is_ok() {
        all_scores.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
        println!("  [best_lattice] top candidates (score, spacing, phase): {:?}", &all_scores[..all_scores.len().min(8)]);
    }
    let max_score = per_s.iter().map(|x| x.0).max().filter(|s| *s >= 2)?;
    // 并列最高分的间距取"窗口内点最大残差最小"者: 兵/炮字形质心向相反方向
    // 偏移时格点只能取两侧折中, 残差最小的间距才能同时罩住两营; 仍平手取更大间距
    let mut tied: Vec<(f32, f32, f32)> = per_s
        .iter()
        .filter(|x| x.0 == max_score)
        .filter_map(|&(_, s, phase)| {
            let ks = snap(s, phase);
            let (_, lo) = windows_of(&ks).iter().max_by_key(|w| w.0).copied()?;
            let minimax = ks
                .iter()
                .zip(coords)
                .filter_map(|(k, &c)| k.filter(|k| (lo..=lo + 4).contains(k)).map(|k| (c - (phase + k as f32 * s)).abs()))
                .fold(0f32, f32::max);
            Some((minimax, s, phase))
        })
        .collect();
    tied.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1)));
    let (_, s, phase) = tied.first().copied()?;
    let ks = snap(s, phase);
    // 得分与最高分相差 8 以内的窗口都作为候选(按得分降序、位置靠上优先):
    // 棋盘外噪声行的窗口可能比真棋盘窗口得分更高, 真伪由上层结合棋子
    // 分类校验选定; 上限 6 个控制组合数
    let mut cands: Vec<(usize, i32)> = windows_of(&ks).into_iter().filter(|(c, _)| c + 8 >= max_score).collect();
    cands.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    cands.truncate(6);
    let windows: Vec<WindowFit> = cands
        .into_iter()
        .map(|(score, lo)| {
            // 跨度校准: 吸附容差宽(0.22 格), 打分选中的间距可能整体偏斜百分之几,
            // 累积到窗口端部把棋子推出容差 -- 用窗口内点端点的实际距离除以
            // 跨度格数得到候选校准 (s2, phase2)。两端点若分属兵/炮(质心反向
            // 偏移 ~0.3 格)校准反而带偏, 因此按"窗口内点最大残差更小者胜"
            // 保守择优, 只有更贴合内点才采用校准
            // 窗口内点按格点分组取坐标中位数(剔除零星混入的噪声内点),
            // 用首末两组的中位数锚定校准间距与相位
            let mut by_k: Vec<(i32, f32)> = Vec::new();
            for k in lo..=lo + 4 {
                let mut group: Vec<f32> = ks
                    .iter()
                    .zip(coords)
                    .filter_map(|(kk, &c)| (*kk == Some(k)).then_some(c))
                    .collect();
                if !group.is_empty() {
                    group.sort_by(|a, b| a.total_cmp(b));
                    by_k.push((k, group[group.len() / 2]));
                }
            }
            let max_res = |(sf, pf): (f32, f32)| {
                by_k
                    .iter()
                    .map(|&(k, c)| (c - (pf + k as f32 * sf)).abs())
                    .fold(0f32, f32::max)
            };
            let calibrated = match (by_k.first(), by_k.last()) {
                (Some(&(kmin, cmin)), Some(&(kmax, cmax))) if kmax > kmin => {
                    let s2 = (cmax - cmin) / (kmax - kmin) as f32;
                    (s2 > 0.0 && (s2 - s).abs() <= s * 0.25)
                        .then(|| (s2, (cmin + cmax) / 2.0 - (kmin + kmax) as f32 / 2.0 * s2))
                }
                _ => None,
            };
            // 保守择优: 校准格点对中位数组的最大残差更小才采用
            let chosen = match calibrated {
                Some(c2) if max_res(c2) < max_res((s, phase)) => c2,
                _ => (s, phase),
            };
            #[cfg(test)]
            if std::env::var("SANPAO_DEBUG").is_ok() {
                println!(
                    "  [calib] lo={lo} s={s:.2} calib={calibrated:?} max_res_orig={:.1} max_res_cal={:.1} chosen={:.2}",
                    max_res((s, phase)),
                    calibrated.map(max_res).unwrap_or(-1.0),
                    chosen.0
                );
            }
            WindowFit {
                score,
                spacing: chosen.0,
                positions: (0..5).map(|i| chosen.1 + (lo + i) as f32 * chosen.0).collect(),
            }
        })
        .collect();
    let top_lo = windows.first().map(|w| (w.positions[4] - phase) / s) // 反解 top 窗口的 lo
        .map(|kmax| kmax as i32 - 4);
    let distinct = match top_lo {
        Some(lo) => {
            let mut used: Vec<i32> = ks.iter().flatten().copied().filter(|k| (lo..=lo + 4).contains(k)).collect();
            used.sort_unstable();
            used.dedup();
            used.len()
        }
        None => 0,
    };
    let spacing = windows.first().map(|w| w.spacing).unwrap_or(s);
    Some(AxisFit { windows, spacing, distinct })
}

// 候选间距来自全部点对距离除以 1..4(真实间距的整倍数必在其中), 去重后交给 best_lattice。
// near=Some(t) 时候选额外约束在 t±15% 内: 一轴被棋盘外噪声带偏时用另一轴间距重拟
fn fit_lattice(coords: &[f32], min_s: f32, max_s: f32, near: Option<f32>) -> Option<AxisFit> {
    let mut cands: Vec<f32> = Vec::new();
    for i in 0..coords.len() {
        for j in i + 1..coords.len() {
            let d = (coords[i] - coords[j]).abs();
            if d < 1e-6 {
                continue;
            }
            for k in 1..=4u32 {
                let s = d / k as f32;
                if s >= min_s && s <= max_s && near.map(|t| (s - t).abs() <= t * 0.15).unwrap_or(true) {
                    cands.push(s);
                }
            }
        }
    }
    cands.sort_by(|a, b| a.total_cmp(b));
    let mut uniq: Vec<f32> = Vec::new();
    for s in cands {
        if uniq.last().map(|u| s - u > *u * 0.03).unwrap_or(true) {
            uniq.push(s);
        }
    }
    best_lattice(coords, &uniq)
}

// 间距已知时只拟相位: 某轴分布退化(不同格点不足 3 个)无法自证间距, 借用另一轴的间距
fn fit_phase(coords: &[f32], s: f32) -> Option<AxisFit> {
    best_lattice(coords, &[s])
}

// 颜色主导性过滤: 棋盘上棋子只有两种颜色, 距两大主色都远的色块是界面噪声
// (按钮/文字颜色五花八门, 且可能成排出现, 以数量优势把等距拟合带偏),
// 标记为不参与格点拟合(仍参与最终格点吸附)。簇心取各通道中位数而非均值,
// 避免混入簇内的少量噪声把簇心拉偏后噪声反被保留
fn color_dominant(blobs: &[Blob]) -> Vec<bool> {
    let n = blobs.len();
    if n < 2 {
        return vec![true; n];
    }
    let dist2 = |a: [f64; 3], b: [f64; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    // 以互相最远的一对色块为初始中心(确定性), Lloyd 迭代至稳定
    let farthest = |c: [f64; 3]| blobs.iter().map(|b| b.rgb).max_by(|a, b| dist2(*a, c).total_cmp(&dist2(*b, c))).unwrap();
    let (mut c0, mut c1) = {
        let c1 = farthest(blobs[0].rgb);
        (farthest(c1), c1)
    };
    let mut assign = vec![0usize; n];
    for _ in 0..12 {
        let mut changed = false;
        for (i, b) in blobs.iter().enumerate() {
            let a = if dist2(b.rgb, c0) <= dist2(b.rgb, c1) { 0 } else { 1 };
            if assign[i] != a {
                assign[i] = a;
                changed = true;
            }
        }
        for (c, center) in [&mut c0, &mut c1].into_iter().enumerate() {
            let mut acc = [0f64; 3];
            let mut cnt = 0usize;
            for (i, b) in blobs.iter().enumerate() {
                if assign[i] == c {
                    acc = add3(acc, b.rgb);
                    cnt += 1;
                }
            }
            if cnt > 0 {
                *center = scale3(acc, 1.0 / cnt as f64);
            }
        }
        if !changed {
            break;
        }
    }
    // 簇心取成员各通道中位数(抗混入簇内的噪声)
    let median3 = |want: usize| -> Option<[f64; 3]> {
        let members: Vec<[f64; 3]> = blobs.iter().enumerate().filter(|&(i, _)| assign[i] == want).map(|(_, b)| b.rgb).collect();
        if members.is_empty() {
            return None;
        }
        let mut out = [0f64; 3];
        for ch in 0..3 {
            let mut v: Vec<f64> = members.iter().map(|m| m[ch]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            out[ch] = v[v.len() / 2];
        }
        Some(out)
    };
    let (Some(m0), Some(m1)) = (median3(0), median3(1)) else {
        return vec![true; n]; // 某簇为空(全同色), 不过滤
    };
    // 距两簇心都超过 0.7 倍簇间距的是噪声色; 下限 40 防止两色相近时误杀棋子
    let thresh = (0.7 * dist2(m0, m1).sqrt()).max(40.0);
    blobs
        .iter()
        .map(|b| dist2(b.rgb, m0).sqrt().min(dist2(b.rgb, m1).sqrt()) <= thresh)
        .collect()
}

// 在盘棋子引用: (列索引, 行索引, 色块)
type PieceRefs<'a> = Vec<(usize, usize, &'a Blob)>;

// 把在盘棋子分成炮/兵两营:
// 字形通道: 红字炮以高饱和像素为主, 黑字兵以墨迹为主; 炮候选按红度 r-g 排序取
//   前 3 个(字形暗红 r-g>=97 高于金色走子标记的 ~95), 多余候选当作标记丢弃;
//   尺寸明显大于中位数的非炮块是"叠了标记的空槽"等装饰, 不计入兵
// 彩色通道: 棋子是两种颜色的实心盘, 按颜色 2 均值聚成两簇, 恰好 3 枚的一簇是炮
fn classify_pieces<'a>(
    pieces: &'a [(usize, usize, &'a Blob)],
    glyph: bool,
    med: f32,
    hint: Option<&ColorHint>,
) -> Result<(PieceRefs<'a>, PieceRefs<'a>), String> {
    // 已知两方颜色(用户点过炮/兵): 按最近色心分类, 免疫换肤/变暗/标记;
    // 炮不足 3 个时不报错, 由上层逐格红像素兜底补齐
    if let Some(h) = hint {
        let dc = |b: &Blob| color_dist(&b.rgb, &h.cannon);
        let ds = |b: &Blob| color_dist(&b.rgb, &h.soldier);
        let mut cands: Vec<(usize, usize, &Blob)> = pieces
            .iter()
            .filter(|&&(_, _, b)| dc(b) < ds(b))
            .map(|&(ix, iy, b)| (ix, iy, b))
            .collect();
        cands.sort_by(|a, b| dc(a.2).total_cmp(&dc(b.2)));
        let cannons: Vec<(usize, usize, &Blob)> = cands.iter().take(3).map(|&(ix, iy, b)| (ix, iy, b)).collect();
        let soldiers: Vec<(usize, usize, &Blob)> = pieces
            .iter()
            .filter(|&&(_, _, b)| ds(b) <= dc(b) && (!glyph || b.area as f32 <= med * 1.8))
            .map(|&(ix, iy, b)| (ix, iy, b))
            .collect();
        if soldiers.is_empty() {
            return Err("未识别到兵方棋子(对局可能已结束), 拟合失败".to_string());
        }
        return Ok((cannons, soldiers));
    }
    if glyph {
        let ratio = |b: &Blob| b.colored as f32 / b.area as f32;
        let mut cands: Vec<(usize, usize, &Blob)> = pieces
            .iter()
            .filter(|&&(_, _, b)| ratio(b) >= 0.5)
            .map(|&(ix, iy, b)| (ix, iy, b))
            .collect();
        // 红度降序: 真炮排在金色走子标记之前; 不足 3 个不在此报错(上层红像素兜底)
        cands.sort_by(|a, b| (b.2.rgb[0] - b.2.rgb[1]).total_cmp(&(a.2.rgb[0] - a.2.rgb[1])));
        let cannons: Vec<(usize, usize, &Blob)> = cands.iter().take(3).map(|&(ix, iy, b)| (ix, iy, b)).collect();
        // 0.6 而非 0.5: 叠加标记的兵墨迹占比可能恰好过半, 宁可放宽不丢兵
        // (放宽误收的块由"炮恒 3"竞争与下面的尺寸过滤排除)
        let soldiers: Vec<(usize, usize, &Blob)> = pieces
            .iter()
            .filter(|&&(_, _, b)| ratio(b) < 0.6 && b.area as f32 <= med * 1.8)
            .map(|&(ix, iy, b)| (ix, iy, b))
            .collect();
        if soldiers.is_empty() {
            return Err("未识别到兵方棋子(对局可能已结束), 拟合失败".to_string());
        }
        Ok((cannons, soldiers))
    } else {
        let rgb = |p: &(usize, usize, &Blob)| -> [f32; 3] { p.2.rgb.map(|v| v as f32) };
        let dist2 = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
        // 2 均值: 以互相最远的一对色块为初始中心(确定性), Lloyd 迭代至稳定
        let farthest = |c: [f32; 3]| rgb(pieces.iter().max_by(|a, b| dist2(rgb(a), c).total_cmp(&dist2(rgb(b), c))).unwrap());
        let (mut c0, mut c1) = {
            let c1 = farthest(rgb(&pieces[0]));
            let c0 = farthest(c1);
            (c0, c1)
        };
        let mut assign = vec![0usize; pieces.len()];
        for _ in 0..12 {
            let mut changed = false;
            for (i, p) in pieces.iter().enumerate() {
                let a = if dist2(rgb(p), c0) <= dist2(rgb(p), c1) { 0 } else { 1 };
                if assign[i] != a {
                    assign[i] = a;
                    changed = true;
                }
            }
            for (c, center) in [&mut c0, &mut c1].into_iter().enumerate() {
                let mut acc = [0f64; 3];
                let mut cnt = 0usize;
                for (i, p) in pieces.iter().enumerate() {
                    if assign[i] == c {
                        acc = add3(acc, p.2.rgb);
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    *center = scale3(acc, 1.0 / cnt as f64).map(|v| v as f32);
                }
            }
            if !changed {
                break;
            }
        }
        let cl0: Vec<(usize, usize, &Blob)> = pieces.iter().enumerate().filter(|&(i, _)| assign[i] == 0).map(|(_, p)| *p).collect();
        let cl1: Vec<(usize, usize, &Blob)> = pieces.iter().enumerate().filter(|&(i, _)| assign[i] == 1).map(|(_, p)| *p).collect();
        // 两簇心必须显著可分: 真平台两色棋子簇心距离远(红/蓝 ~200);
        // 杂色噪声块(木色调界面)强分出来的两簇几乎同色, 是误报
        let sep = dist2(c0, c1).sqrt();
        if sep < 60.0 {
            return Err(format!("两簇棋子颜色几乎相同(簇心距 {sep:.0}), 不是双色棋盘, 拟合失败"));
        }
        match (cl0.len(), cl1.len()) {
            (3, 3) => Err("炮兵两簇各 3 枚, 无法区分阵营, 拟合失败".to_string()),
            (3, _) => Ok((cl0, cl1)),
            (_, 3) => Ok((cl1, cl0)),
            (a, b) => Err(format!("无法按颜色区分炮/兵(两簇 {a}/{b} 枚, 炮簇须恰为 3), 拟合失败")),
        }
    }
}

// 启动标注的颜色提示: 自动识别失败时, 用户在首帧截图上点一门炮/一枚兵,
// 采样两方真实颜色。已知颜色后分类免疫换肤/变暗/高亮标记/笔画过细
#[derive(Clone, Copy, Debug)]
pub struct ColorHint {
    pub cannon: [f32; 3],
    pub soldier: [f32; 3],
}

// 在点击处采样一方颜色: 7x7 窗口内"像棋子"的像素(深色或高饱和)取各通道中位数
pub fn sample_camp_color(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, x: f32, y: f32) -> Option<[f32; 3]> {
    let (w, h) = img.dimensions();
    let mut chans: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for dy in -3i64..=3 {
        for dx in -3i64..=3 {
            let (px, py) = (x as i64 + dx, y as i64 + dy);
            if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                continue;
            }
            let p = img.get_pixel(px as u32, py as u32);
            let maxc = p[0].max(p[1]).max(p[2]);
            let minc = p[0].min(p[1]).min(p[2]);
            if maxc < 170 || maxc - minc > SAT_MIN {
                for c in 0..3 {
                    chans[c].push(p[c]);
                }
            }
        }
    }
    if chans[0].is_empty() {
        return None;
    }
    let med = |v: &mut Vec<u8>| {
        v.sort_unstable();
        v[v.len() / 2] as f32
    };
    Some([med(&mut chans[0]), med(&mut chans[1]), med(&mut chans[2])])
}

// 逐格点数"贴近炮色"的像素找炮位(兜底): 缩放后的窗口炮字笔画过细,
// 腐蚀通道提取不到完整炮字块(只剩 0-2 个炮色块)时使用。
// 实测炮字鲜红(r≈194-212)、空槽暗红(r≈127)、金色标记 r≈165, 按颜色距离
// 区分三者; 完整对中的炮字红像素约 0.078*格距^2, 格点偏到一半只剩 ~0.02 --
// 阈值取 0.04*格距^2(下限 80), 偏心的炮位(错间距格点的标志)不凑数
const RED_CELL_FRAC: f32 = 0.04;
const RED_CELL_MIN: usize = 80;
const RED_CELL_RADIUS: f32 = 60.0;

// 红像素炮位扫描结果: (恰好 3 个炮位格点, 红像素均色)
type RedCellHit = (Vec<(usize, usize)>, [f32; 3]);

fn red_cell_cannons(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    xs: &[f32],
    ys: &[f32],
    cell: f32,
    cannon_ref: [f32; 3],
) -> Option<RedCellHit> {
    let (w, h) = img.dimensions();
    let r = (cell * 0.35) as i32;
    let min_count = ((cell * cell * RED_CELL_FRAC) as usize).max(RED_CELL_MIN);
    let mut hits: Vec<(usize, usize, usize, [f64; 3])> = Vec::new();
    for (iy, &cy) in ys.iter().enumerate() {
        for (ix, &cx) in xs.iter().enumerate() {
            let mut count = 0usize;
            let mut sum = [0f64; 3];
            for dy in -r..=r {
                for dx in -r..=r {
                    let (px, py) = (cx as i32 + dx, cy as i32 + dy);
                    if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                        continue;
                    }
                    let p = img.get_pixel(px as u32, py as u32);
                    let d = ((p[0] as f32 - cannon_ref[0]).powi(2)
                        + (p[1] as f32 - cannon_ref[1]).powi(2)
                        + (p[2] as f32 - cannon_ref[2]).powi(2))
                        .sqrt();
                    if d < RED_CELL_RADIUS {
                        count += 1;
                        sum = add3(sum, [p[0] as f64, p[1] as f64, p[2] as f64]);
                    }
                }
            }
            if count >= min_count {
                hits.push((ix, iy, count, sum));
            }
            #[cfg(test)]
            if std::env::var("SANPAO_DEBUG").is_ok() && count > 0 {
                println!("  [red_cell] cell({ix},{iy}) at ({cx:.0},{cy:.0}) count={count}");
            }
        }
    }
    // 取红密度最大的 3 个格点为炮位, 但要求对第 4 名有压倒性优势
    // (真炮字 ~650 px, 恰好套在格点上的红色界面元素 ~120 px):
    // 优势不足说明窗口套偏到了界面上, 宁可失败交由上层换窗口/报错
    hits.sort_by_key(|h| std::cmp::Reverse(h.2));
    if hits.len() < 3 || (hits.len() > 3 && hits[3].2 * 3 >= hits[2].2) {
        return None;
    }
    let hits: Vec<(usize, usize, usize, [f64; 3])> = hits.into_iter().take(3).collect();
    let mut sum = [0f64; 3];
    let mut n = 0usize;
    for (_, _, c, s) in &hits {
        sum = add3(sum, *s);
        n += c;
    }
    let mean = scale3(sum, 1.0 / n.max(1) as f64).map(|v| v as f32);
    Some((hits.into_iter().map(|(ix, iy, _, _)| (ix, iy)).collect(), mean))
}

// 由任意局面自举: 对色块 x/y 各做一维等距格点拟合得到 5x5 网格, 吸附到格点的
// 非装饰色块按颜色分为炮/兵, 由此建立网格模型与两方颜色质心。
// 监听启动时对局已在进行中、无法回到开局的场景走这条路(开局指纹失败后兜底)。
// 已知局限: 端部整行/列空出时网格端点有平移歧义 -- 行按"空端行在炮侧"推断
// (炮的家在边行, 全部离家后家行空出; 兵推进致自家远端空出时会推错),
// 列默认靠左; 推错由"纠"按钮兜底(标注局面与识别恰为整体平移时, 网格随之平移)
fn fit_arbitrary(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    blobs: &[Blob],
    step: usize,
    glyph: bool,
    eroded: bool,
    hint: Option<&ColorHint>,
) -> Result<FirstDetect, String> {
    if blobs.len() < 4 {
        return Err(format!("识别到 {} 个色块, 至少需要 4 个(3 炮 + 存活兵), 请确保棋盘完整可见", blobs.len()));
    }
    let (iw, ih) = img.dimensions();
    let scale = step as f32;
    let med = median_area(blobs);
    // 间距下界: 棋子等价直径的 1.1 倍(相邻棋子中心距即格距); 上界: 棋盘 5 格须在画面内
    let min_s = (1.1 * 2.0 * (med / std::f32::consts::PI).sqrt()).max(8.0);
    // 格点拟合只用两大主色的色块(排除成排的界面噪声), 吸附阶段再用全部色块;
    // 用户点过炮/兵时直接按已知颜色过滤(比 2 均值推断可靠)
    let keep = match hint {
        Some(h) => {
            let sep = color_dist(&h.cannon.map(|v| v as f64), &h.soldier).sqrt().max(1.0);
            let thresh = sep * 0.45;
            blobs
                .iter()
                // color_dist 返回平方距离, 比较前开方
                .map(|b| color_dist(&b.rgb, &h.cannon).min(color_dist(&b.rgb, &h.soldier)).sqrt() <= thresh)
                .collect()
        }
        None => color_dominant(blobs),
    };
    #[cfg(test)]
    if let (true, Some(h)) = (std::env::var("SANPAO_DEBUG").is_ok(), hint) {
        let sep = color_dist(&h.cannon.map(|v| v as f64), &h.soldier).sqrt().max(1.0);
        println!(
            "  [fit_arbitrary] hint=炮({:.0},{:.0},{:.0}) 兵({:.0},{:.0},{:.0}) sep={sep:.0} keep={}/{}",
            h.cannon[0], h.cannon[1], h.cannon[2],
            h.soldier[0], h.soldier[1], h.soldier[2],
            keep.iter().filter(|k| **k).count(),
            keep.len()
        );
        for (i, b) in blobs.iter().enumerate().take(8) {
            println!(
                "  [blob{i}] ({:.0},{:.0}) dc={:.0} ds={:.0} keep={} rgb=({:.0},{:.0},{:.0})",
                b.cx, b.cy,
                color_dist(&b.rgb, &h.cannon).sqrt(),
                color_dist(&b.rgb, &h.soldier).sqrt(),
                keep[i], b.rgb[0], b.rgb[1], b.rgb[2]
            );
        }
    }
    let xs: Vec<f32> = blobs.iter().zip(&keep).filter(|(_, k)| **k).map(|(b, _)| b.cx).collect();
    let ys: Vec<f32> = blobs.iter().zip(&keep).filter(|(_, k)| **k).map(|(b, _)| b.cy).collect();
    let fx = fit_lattice(&xs, min_s, iw as f32 / scale / 4.5, None);
    let fy = fit_lattice(&ys, min_s, ih as f32 / scale / 4.5, None);
    // 每轴至少 3 个不同格点才能自证间距; 一轴退化时借用另一轴的间距只拟相位
    let (fx, fy) = match (&fx, &fy) {
        (Some(a), Some(b)) if a.distinct >= 3 && b.distinct >= 3 => {
            let (sa, sb) = (a.spacing, b.spacing);
            if (sa - sb).abs() <= sa.max(sb) * 0.15 {
                (fx.unwrap(), fy.unwrap())
            } else {
                // 棋盘是正方形网格且无透视, 两轴间距差应在 15% 内(真帧 6% 内);
                // 不一致说明一轴被棋盘外噪声行带偏 -- 用另一轴的间距约束重拟
                if let Some(fy2) = fit_lattice(&ys, min_s, ih as f32 / scale / 4.5, Some(sa)).filter(|f| f.distinct >= 3) {
                    (fx.unwrap(), fy2)
                } else if let Some(fx2) = fit_lattice(&xs, min_s, iw as f32 / scale / 4.5, Some(sb)).filter(|f| f.distinct >= 3) {
                    (fx2, fy.unwrap())
                } else {
                    return Err("棋盘横纵间距不一致, 任意局面拟合失败".to_string());
                }
            }
        }
        (Some(a), None) if a.distinct >= 3 => {
            let fy = fit_phase(&ys, a.spacing).ok_or("行方向棋子分布退化, 拟合失败")?;
            (fx.unwrap(), fy)
        }
        (None, Some(b)) if b.distinct >= 3 => {
            let fx = fit_phase(&xs, b.spacing).ok_or("列方向棋子分布退化, 拟合失败")?;
            (fx, fy.unwrap())
        }
        _ => return Err("棋子分布过于稀疏, 无法定位棋盘网格(需分布在至少 3 行和 3 列)".to_string()),
    };

    // 窗口候选逐组合尝试: 吸附到格点 -> 分类 -> 数量校验全部通过才算成功。
    // 棋盘外的装饰行/端部整行空出都可能让得分最高的窗口不是真棋盘,
    // 错误窗口会因"炮候选凑不齐 3 个"或分类数量异常被淘汰
    let deco = make_deco_filter(glyph, blobs);
    let attempt = |xpos: &[f32], ypos: &[f32]| -> Result<FirstDetect, String> {
        // 吸附到格点(两轴都距格点 SNAP_FRAC 格内)的主色、非装饰色块为在盘棋子
        let snap_index = |positions: &[f32], spacing: f32, c: f32| -> Option<usize> {
            let i = nearest_index(positions, c);
            ((c - positions[i]).abs() <= spacing * SNAP_FRAC).then_some(i)
        };
        let pieces: Vec<(usize, usize, &Blob)> = blobs
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .filter_map(|(b, _)| {
                Some((
                    snap_index(xpos, fx.spacing, b.cx)?,
                    snap_index(ypos, fy.spacing, b.cy)?,
                    b,
                ))
            })
            .filter(|(_, _, b)| !deco(b))
            .collect();
        #[cfg(test)]
        if std::env::var("SANPAO_DEBUG").is_ok() {
            println!(
                "  [fit_arbitrary] xs={xpos:?} ys={ypos:?} sx={:.1} sy={:.1} keep={}",
                fx.spacing,
                fy.spacing,
                keep.iter().filter(|k| **k).count()
            );
            for (i, b) in blobs.iter().enumerate() {
                println!(
                    "  [blob{}] ({:.0},{:.0}) area={} colored={} ix={:?} iy={:?} deco={} rgb=({:.0},{:.0},{:.0})",
                    i, b.cx, b.cy, b.area, b.colored,
                    if keep[i] { snap_index(xpos, fx.spacing, b.cx) } else { None },
                    if keep[i] { snap_index(ypos, fy.spacing, b.cy) } else { None },
                    deco(b), b.rgb[0], b.rgb[1], b.rgb[2]
                );
            }
            println!("  [fit_arbitrary] pieces={}", pieces.len());
        }
        if pieces.len() < 4 {
            return Err(format!("吸附到网格的棋子仅 {} 个(至少 3 炮 + 1 兵), 拟合失败", pieces.len()));
        }
        let (cannon_blobs, soldiers) = classify_pieces(&pieces, glyph, med, hint)?;
        let mean_color = |ps: &[(usize, usize, &Blob)]| -> [f32; 3] {
            let mut acc = [0f64; 3];
            for p in ps {
                acc = add3(acc, p.2.rgb);
            }
            scale3(acc, 1.0 / ps.len() as f64).map(|v| v as f32)
        };
        let cell = (fx.spacing + fy.spacing) / 2.0;
        // 炮位与炮色: 色块分类优先; 不足 3 个(缩放后炮字太细提取不到完整块)时
        // 逐格红像素密度兜底, 参考色用用户标注的炮色或平台实测红
        let (cannon_cells, cannon_color) = if cannon_blobs.len() == 3 {
            (cannon_blobs.iter().map(|&(ix, iy, _)| (ix, iy)).collect::<Vec<_>>(), mean_color(&cannon_blobs))
        } else {
            let reference = hint.map(|h| h.cannon).unwrap_or([205.0, 8.0, 5.0]);
            let xs_raw: Vec<f32> = xpos.iter().map(|v| v * scale).collect();
            let ys_raw: Vec<f32> = ypos.iter().map(|v| v * scale).collect();
            let scan = red_cell_cannons(img, &xs_raw, &ys_raw, cell * scale, reference);
            #[cfg(test)]
            if std::env::var("SANPAO_DEBUG").is_ok() {
                println!("  [red_cell] ref={reference:?} cell={:.1} xs={xs_raw:?} ys={ys_raw:?} -> {:?}", cell * scale, scan.as_ref().map(|(c, _)| c.clone()));
            }
            match scan {
                Some((cells, red_mean)) => (cells, red_mean),
                None => return Err(format!("炮的候选仅 {} 个(需 3 个), 拟合失败", cannon_blobs.len())),
            }
        };

        // 端部空行的归属: 空端行在炮所在一侧; 列无类似信号, 默认靠左。
        // 棋子块占窗口相对行 [y_min, y_max], 整体放入棋盘时以 y_min 为基准平移
        let mean_y = |ps: &[(usize, usize, &Blob)]| ps.iter().map(|p| p.1 as f32).sum::<f32>() / ps.len() as f32;
        let cannon_mean_y = cannon_cells.iter().map(|c| c.1 as f32).sum::<f32>() / cannon_cells.len() as f32;
        let (y_min, y_max) = {
            let mut lo = usize::MAX;
            let mut hi = 0;
            for p in pieces.iter().map(|p| p.1).chain(cannon_cells.iter().map(|c| c.1)) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
            (lo, hi)
        };
        let offset_y = if y_max - y_min < 4 && cannon_mean_y + 0.3 < mean_y(&soldiers) {
            4 - (y_max - y_min)
        } else {
            0
        };

        // 先放兵再放炮: 兜底炮位来自格点红像素, 覆盖该格上误分类的杂色块
        let mut board = [[EMPTY; game::SIZE]; game::SIZE];
        for &(ix, iy, _) in &soldiers {
            board[iy - y_min + offset_y][ix] = SOLDIER;
        }
        for &(ix, iy) in &cannon_cells {
            board[iy - y_min + offset_y][ix] = CANNON;
        }
        let (c, s) = game::counts(&board);
        // 兵至少 2 枚: 只剩 1 枚兵的对局一步就终局, 而错误窗口拼出的伪局面
        // 常恰好只有 1 枚"兵"
        if c != 3 || !(2..=15).contains(&s) {
            return Err(format!("棋子分类异常(炮 {c} 兵 {s}), 拟合失败"));
        }

        let cannons_top = cannon_mean_y < mean_y(&soldiers);
        // 裁剪框比开局路径外扩到 1.8 格(开局是 0.8): 端部行列归属可能有误,
        // "纠"按钮平移对齐后新端点行/列仍须在裁剪框内
        let x0 = ((xpos[0] - fx.spacing * 1.8).max(0.0) * scale) as u32;
        let y0 = ((ypos[0] - fy.spacing * 1.8).max(0.0) * scale) as u32;
        let x1 = (((xpos[4] + fx.spacing * 1.8) * scale) as u32).min(iw);
        let y1 = (((ypos[4] + fy.spacing * 1.8) * scale) as u32).min(ih);
        let crop = (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0));
        let model = GridModel {
            xs: xpos.iter().map(|x| x * scale - x0 as f32).collect(),
            ys: ypos.iter().map(|y| y * scale - y0 as f32).collect(),
            cell: cell * scale,
            cannon_color,
            soldier_color: mean_color(&soldiers),
            cannons_top,
            eroded,
            glyph,
        };
        Ok(FirstDetect { board, model, crop })
    };
    // 全部组合评完择优: 错误窗口可能凑出恰好合法的伪局面(3 炮 + 少量兵),
    // 但真棋盘会吸附住所有在盘棋子 -- 先比吸附的棋格数(解释力), 再比窗口得分
    let mut best_pass: Option<((usize, usize), FirstDetect)> = None;
    let mut last_err = String::new();
    for wy in &fy.windows {
        for wx in &fx.windows {
            let combo_score = wy.score + wx.score;
            if best_pass.as_ref().is_some_and(|(k, _)| (0, combo_score) <= *k) {
                continue; // 棋格数上限已定且窗口得分不可能更高
            }
            match attempt(&wx.positions, &wy.positions) {
                Ok(det) => {
                    let (c, s) = game::counts(&det.board);
                    let key = (c + s, combo_score);
                    #[cfg(test)]
                    if std::env::var("SANPAO_DEBUG").is_ok() {
                        let rows: Vec<String> = det.board.iter().map(|r| r.iter().collect()).collect();
                        println!("  [combo PASS key={key:?}] {}", rows.join("/"));
                    }
                    if best_pass.as_ref().is_none_or(|(k, _)| key > *k) {
                        best_pass = Some((key, det));
                    }
                }
                Err(e) => {
                    #[cfg(test)]
                    if std::env::var("SANPAO_DEBUG").is_ok() {
                        println!("  [combo score={combo_score}] {e}");
                    }
                    if last_err.is_empty() {
                        last_err = e;
                    }
                }
            }
        }
    }
    if let Some((_, det)) = best_pass {
        return Ok(det);
    }
    Err(if last_err.is_empty() {
        "未产生任何网格窗口候选, 拟合失败".to_string()
    } else {
        last_err
    })
}


// 平台会在刚走动的棋子/出发格上叠加红金色调标记(落在红字掩码内), 因此:
// 1. 炮恒为 3: 炮色候选全局只保留颜色最贴近炮质心的 3 个, 多余的视为标记;
// 2. 标记格点的归属: 含像兵的色块、或深色墨迹占多数(字形本底还在, 只是均色被
//    标记拉偏)判为兵; 几乎纯彩色(纯标记/空槽上的装饰)才判空
pub fn detect_frame(model: &GridModel, img: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Option<Board> {
    let erode = if model.eroded { 2 } else { 0 };
    let (mut blobs, _step) = extract_blobs(img, 640, erode, model.glyph);
    // 按面积降序: 大色块优先, 移动动画的残影等小块随后被拒
    blobs.sort_by(|a, b| b.area.cmp(&a.area));
    let deco = make_deco_filter(model.glyph, &blobs);

    // 收集吸附到每个格点的全部色块(同一格可能同时有棋子字形和标记)
    let mut points: [[Vec<&Blob>; game::SIZE]; game::SIZE] = Default::default();
    for b in blobs.iter() {
        if deco(b) {
            continue;
        }
        let ix = nearest_index(&model.xs, b.cx);
        let iy = nearest_index(&model.ys, b.cy);
        let dx = b.cx - model.xs[ix];
        let dy = b.cy - model.ys[iy];
        if dx * dx + dy * dy > (model.cell * 0.4) * (model.cell * 0.4) {
            continue; // 不在网格点附近: 移动动画或噪声
        }
        points[iy][ix].push(b);
    }

    // 炮色候选: 全局取颜色最贴近炮质心的 3 个, 不足 3 则本帧作废等待下一帧
    let mut cannon_cands: Vec<(usize, usize, f32)> = Vec::new();
    for (iy, row) in points.iter().enumerate() {
        for (ix, pb) in row.iter().enumerate() {
            for b in pb {
                // 墨迹占多数的色块是"被标记的兵"(兵字本底还在, 只是均色被红金标记拉偏),
                // 无条件不得参与炮位竞争 -- 防止它在真炮被高亮/动画遮蔽的瞬间顶替成炮
                if (b.colored as f32) < b.area as f32 * 0.5 {
                    continue;
                }
                if color_dist(&b.rgb, &model.cannon_color) < color_dist(&b.rgb, &model.soldier_color) {
                    cannon_cands.push((ix, iy, color_dist(&b.rgb, &model.cannon_color)));
                }
            }
        }
    }
    // 炮色候选: 全局取颜色最贴近炮质心的 3 个; 不足 3 个(缩放后炮字太细
    // 提取不到完整块)时按格点红像素密度兜底, 仍不足则本帧作废等待下一帧
    let cannon_pts: Vec<(usize, usize)> = if cannon_cands.len() >= 3 {
        cannon_cands.sort_by(|a, b| a.2.total_cmp(&b.2));
        cannon_cands.iter().take(3).map(|&(x, y, _)| (x, y)).collect()
    } else {
        red_cell_cannons(img, &model.xs, &model.ys, model.cell, model.cannon_color)?.0
    };

    let mut board = [[EMPTY; game::SIZE]; game::SIZE];
    for (iy, row) in points.iter().enumerate() {
        for (ix, pb) in row.iter().enumerate() {
            if cannon_pts.contains(&(ix, iy)) {
                board[iy][ix] = CANNON;
                continue;
            }
            if pb.is_empty() {
                continue;
            }
            let soldier_like = pb.iter().any(|b| {
                color_dist(&b.rgb, &model.soldier_color) < color_dist(&b.rgb, &model.cannon_color)
                    || (b.colored as f32) < b.area as f32 * 0.5
            });
            if soldier_like {
                board[iy][ix] = SOLDIER;
            }
        }
    }
    Some(board)
}


// 色块面积中位数(棋子尺寸的典型值)
fn median_area(blobs: &[Blob]) -> f32 {
    let mut areas: Vec<usize> = blobs.iter().map(|b| b.area).collect();
    areas.sort_unstable();
    areas.get(areas.len() / 2).map(|v| *v as f32).unwrap_or(f32::MAX)
}

// 装饰块过滤(字形通道): 棋子字形尺寸一致, 而炮位空槽等装饰是无彩深色且明显更大
fn make_deco_filter(glyph: bool, blobs: &[Blob]) -> impl Fn(&Blob) -> bool + '_ {
    let med = median_area(blobs);
    move |b: &Blob| glyph && b.colored == 0 && b.area as f32 > med * 1.8
}

// 手动纠错: 用用户在镜像棋盘上标注的局面重校准两方颜色质心
// 逐个标注格找吸附到该格的最大色块(字形整体而非碎片), 按标注阵营累计取均值
pub fn calibrate_colors(model: &GridModel, img: &ImageBuffer<Rgba<u8>, Vec<u8>>, board: &Board) -> Option<([f32; 3], [f32; 3])> {
    let erode = if model.eroded { 2 } else { 0 };
    let (blobs, _) = extract_blobs(img, 640, erode, model.glyph);
    let snap2 = model.cell * 0.4 * (model.cell * 0.4);
    let mut cannon = ([0f64; 3], 0usize);
    let mut soldier = ([0f64; 3], 0usize);
    for y in 0..game::SIZE {
        for x in 0..game::SIZE {
            let piece = board[y][x];
            if piece == EMPTY {
                continue;
            }
            let px = model.xs[x];
            let py = model.ys[y];
            let best = blobs
                .iter()
                .filter(|b| {
                    let dx = b.cx - px;
                    let dy = b.cy - py;
                    dx * dx + dy * dy <= snap2
                })
                .max_by(|a, b| a.area.cmp(&b.area));
            if let Some(b) = best {
                if piece == CANNON {
                    cannon.0 = add3(cannon.0, b.rgb);
                    cannon.1 += 1;
                } else {
                    soldier.0 = add3(soldier.0, b.rgb);
                    soldier.1 += 1;
                }
            }
        }
    }
    if cannon.1 == 0 || soldier.1 == 0 {
        return None; // 某一方没采到样本, 保留旧质心
    }
    let cc = scale3(cannon.0, 1.0 / cannon.1 as f64).map(|v| v as f32);
    let sc = scale3(soldier.0, 1.0 / soldier.1 as f64).map(|v| v as f32);
    Some((cc, sc))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{pos_of, xy_of};
    use xcap::image::GenericImageView;

    // 合成棋盘截图: 木色背景 + 网格线 + 红/蓝圆形棋子
    // line_sat 为 true 时网格线用高饱和棕色(会被掩码且穿过棋子中心), 模拟真实平台
    fn synth(board: &Board, origin: (u32, u32), cell: u32, noise: u64, line_sat: bool) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let w = origin.0 + cell * 6 + 40;
        let h = origin.1 + cell * 6 + 40;
        let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(w, h);
        let mut rng = noise;
        let mut next = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) % 8) as u8
        };
        for y in 0..h {
            for x in 0..w {
                let n = next();
                let p = img.get_pixel_mut(x, y);
                p[0] = 206 + n; // 木色低饱和背景
                p[1] = 178 + n;
                p[2] = 140 + n;
                p[3] = 255;
            }
        }
        // 网格线画在交叉点上(穿过棋子中心)
        let line: [u8; 3] = if line_sat { [138, 90, 43] } else { [90, 60, 30] };
        for i in 0..5 {
            let px = origin.0 + cell * i as u32 + cell / 2;
            let py = origin.1 + cell * i as u32 + cell / 2;
            for t in 0..cell * 4 {
                let x = origin.0 + cell / 2 + t;
                if x < w && py < h {
                    let p = img.get_pixel_mut(x, py);
                    p[0] = line[0];
                    p[1] = line[1];
                    p[2] = line[2];
                }
                let y = origin.1 + cell / 2 + t;
                if y < h && px < w {
                    let p = img.get_pixel_mut(px, y);
                    p[0] = line[0];
                    p[1] = line[1];
                    p[2] = line[2];
                }
            }
        }
        // 棋子: 炮红 兵蓝
        for y in 0..game::SIZE {
            for x in 0..game::SIZE {
                let piece = board[y][x];
                if piece == EMPTY {
                    continue;
                }
                let cx = origin.0 + cell * x as u32 + cell / 2;
                let cy = origin.1 + cell * y as u32 + cell / 2;
                let r = (cell as f32 * 0.38) as i32;
                let color: [u8; 3] = if piece == CANNON { [198, 62, 52] } else { [62, 92, 198] };
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy <= r * r {
                            let px = cx as i32 + dx;
                            let py = cy as i32 + dy;
                            if px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                                let p = img.get_pixel_mut(px as u32, py as u32);
                                p[0] = color[0];
                                p[1] = color[1];
                                p[2] = color[2];
                            }
                        }
                    }
                }
            }
        }
        img
    }

    fn crop_image(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, crop: (u32, u32, u32, u32)) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let view = img.view(crop.0, crop.1, crop.2, crop.3);
        let mut out = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(crop.2, crop.3);
        for y in 0..crop.3 {
            for x in 0..crop.2 {
                out.put_pixel(x, y, view.get_pixel(x, y));
            }
        }
        out
    }

    #[test]
    fn test_detect_first_startpos() {
        let board = game::startpos_board();
        let img = synth(&board, (57, 93), 41, 42, false);
        let det = detect_first(&img, None).unwrap();
        assert_eq!(det.board, game::startpos_board());
        assert!(det.model.cannons_top);
        assert!(!det.model.eroded, "低饱和线条应走主通道");
        assert!(det.crop.2 > 0 && det.crop.3 > 0);
        for x in &det.model.xs {
            assert!(*x >= 0.0 && *x <= det.crop.2 as f32, "x={x} 超出裁剪框宽 {}", det.crop.2);
        }
        for y in &det.model.ys {
            assert!(*y >= 0.0 && *y <= det.crop.3 as f32, "y={y} 超出裁剪框高 {}", det.crop.3);
        }
    }

    #[test]
    fn test_detect_first_saturated_lines() {
        // 高饱和网格线穿过棋子中心: 主通道会连片失败, 应自动走腐蚀通道
        let board = game::startpos_board();
        let img = synth(&board, (57, 93), 41, 42, true);
        let det = detect_first(&img, None).unwrap();
        assert_eq!(det.board, game::startpos_board());
        assert!(det.model.eroded, "高饱和线条应触发腐蚀通道");
        // 中局跟踪也应正常
        let next = game::apply_move(&board, "a0a2");
        let next = game::apply_move(&next, "b2b1");
        let img2 = synth(&next, (57, 93), 41, 99, true);
        let crop_img = crop_image(&img2, det.crop);
        assert_eq!(detect_frame(&det.model, &crop_img).unwrap(), next);
    }

    #[test]
    fn test_detect_first_flipped() {
        // 炮在下方的开局(对手执炮视角)
        let board = flipped_startpos();
        let img = synth(&board, (30, 30), 37, 7, false);
        let det = detect_first(&img, None).unwrap();
        assert_eq!(det.board, flipped_startpos());
        assert!(!det.model.cannons_top);
    }

    #[test]
    fn test_detect_frame_midgame() {
        let board = game::startpos_board();
        let img = synth(&board, (57, 93), 41, 42, false);
        let det = detect_first(&img, None).unwrap();
        // 炮隔空吃兵 + 兵上移一步后的局面
        let next = game::apply_move(&board, "a0a2");
        let next = game::apply_move(&next, "b2b1");
        let img2 = synth(&next, (57, 93), 41, 99, false);
        let crop_img = crop_image(&img2, det.crop);
        let detected = detect_frame(&det.model, &crop_img).unwrap();
        assert_eq!(detected, next);
    }

    #[test]
    fn test_detect_frame_cannon_invariant() {
        let board = game::startpos_board();
        let img = synth(&board, (57, 93), 41, 42, false);
        let det = detect_first(&img, None).unwrap();
        // 4 门炮违反"炮恒为 3"不变量的帧必须被拒或纠正为 3 门
        let mut four = board;
        four[0][1] = CANNON;
        let img2 = synth(&four, (57, 93), 41, 5, false);
        let crop_img = crop_image(&img2, det.crop);
        if let Some(b) = detect_frame(&det.model, &crop_img) {
            let (c, _) = game::counts(&b);
            assert_eq!(c, 3, "炮数必须为 3");
        }
    }

    #[test]
    fn test_detect_first_real_browser_screenshot() {
        // 用 tools/testboard.html 的真实浏览器截图做端到端识别验证
        // (截图由浏览器截屏保存在 target/testboard_real.png, 不存在时跳过)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/testboard_real.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("真实浏览器截图识别失败: {e}"));
        assert_eq!(det.board, game::startpos_board());
        assert!(det.model.cannons_top);
    }

    #[test]
    fn test_detect_frame_real_browser_screenshot() {
        // 真实浏览器里走了 "炮 a0 吃 a3" 后的截图, 验证中局跟踪
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/target/testboard_real.png");
        let moved = concat!(env!("CARGO_MANIFEST_DIR"), "/target/testboard_real2.png");
        let (Ok(r1), Ok(r2)) = (
            xcap::image::ImageReader::open(base),
            xcap::image::ImageReader::open(moved),
        ) else {
            return;
        };
        let img1 = r1.decode().expect("解码开局截图失败").to_rgba8();
        let img2 = r2.decode().expect("解码中局截图失败").to_rgba8();
        let det = detect_first(&img1, None).unwrap_or_else(|e| panic!("开局截图识别失败: {e}"));
        let cropped = crop_image(&img2, det.crop);
        let detected = detect_frame(&det.model, &cropped)
            .unwrap_or_else(|| panic!("中局截图识别失败"));
        // 测试页里用 JS 把炮从 a0 拖到 a3(不走规则引擎), 期望局面手工构造
        let mut expect = game::startpos_board();
        expect[0][0] = game::EMPTY;
        expect[3][0] = game::CANNON;
        assert_eq!(detected, expect);
    }

    // 微信小程序风格合成图(按真实截图渲染建模):
    // 饱和橙红底 + 黑色网格线 + 兵=橙面木盘黑字 + 炮=橙面木盘红字(中间三列, 两端空槽),
    // 周围有红色横幅/头像/红色状态文字/底部按钮条等界面噪声
    fn synth_wechat(board: &Board, origin: (u32, u32), cell: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let w = origin.0 + cell * 6 + 60;
        let h = origin.1 + cell * 6 + 170;
        let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(w, h);
        let put = |img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, x: i64, y: i64, c: [u8; 3]| {
            if x >= 0 && y >= 0 && (x as u64) < w as u64 && (y as u64) < h as u64 {
                let p = img.get_pixel_mut(x as u32, y as u32);
                p[0] = c[0];
                p[1] = c[1];
                p[2] = c[2];
            }
        };
        // 全窗饱和橙红底(字形通道不掩码; 彩色通道会整体连成巨型块而失败)
        for y in 0..h {
            for x in 0..w {
                put(&mut img, x as i64, y as i64, [190, 141, 70]);
            }
        }
        // 顶部红色横幅(全宽, 应被长宽比过滤剔除)
        for y in 10..80 {
            for x in 0..w {
                put(&mut img, x as i64, y as i64, [180, 50, 50]);
            }
        }
        // 黑色网格线(3px, 交叉点在 origin + i*cell)
        for i in 0..5 {
            let px = (origin.0 + cell * i) as i64;
            let py = (origin.1 + cell * i) as i64;
            for t in -10..(cell * 4 + 10) as i64 {
                for d in -1..=1 {
                    put(&mut img, origin.0 as i64 + t, py + d, [13, 2, 1]);
                    put(&mut img, px + d, origin.1 as i64 + t, [13, 2, 1]);
                }
            }
        }
        // 头像噪声(深灰圆, 棋盘上方一格右侧)
        let avx = (origin.0 + cell * 4 + 20) as i64;
        let avy = (origin.1 - cell) as i64;
        for dy in -14i64..=14 {
            for dx in -14i64..=14 {
                if dx * dx + dy * dy <= 196 {
                    put(&mut img, avx + dx, avy + dy, [60, 58, 56]);
                }
            }
        }
        // 红色状态文字噪声(一行 5 个小红块, 棋盘上方一格左侧)
        for i in 0..5 {
            let bx = (origin.0 + i * (cell / 3)) as i64;
            let by = (origin.1 - cell) as i64;
            for y in by..by + 12 {
                for x in bx..bx + 12 {
                    put(&mut img, x, y, [204, 53, 36]);
                }
            }
        }
        // 炮行的空位画深色空槽(炮离开后露出, 约 2 倍字形大小), 应被装饰过滤剔除
        let mut cannon_rows: Vec<usize> = (0..game::SIZE)
            .filter(|&y| (0..game::SIZE).any(|x| board[y][x] == CANNON))
            .collect();
        cannon_rows.dedup();
        if cannon_rows.len() == 1 {
            let r = cannon_rows[0];
            let ry = (origin.1 + cell * r as u32) as i64;
            for col in 0..game::SIZE {
                if board[r][col] == EMPTY {
                    let cx = (origin.0 + cell * col as u32) as i64;
                    for dy in -(cell as i64 / 8)..=(cell as i64 / 8) {
                        for dx in -(cell as i64 * 3 / 20)..=(cell as i64 * 3 / 20) {
                            put(&mut img, cx + dx, ry + dy, [80, 70, 60]);
                        }
                    }
                }
            }
        }
        // 中局时在炮的出发格画金色标记(模拟平台走子指示, 颜色落在红字掩码内),
        // 应被"炮恒 3"约束当作标记丢弃。画成十字而非实心块(实心块会被填充率
        // 过滤直接剔除, 到不了炮位竞争, 测不出标记干扰)
        if cannon_rows.len() != 1 && board[4][2] == EMPTY {
            let mx = (origin.0 + cell * 2) as i64;
            let my = (origin.1 + cell * 4) as i64;
            for dy in -6i64..=6 {
                for dx in -6i64..=6 {
                    if dx.abs() <= 1 || dy.abs() <= 1 {
                        put(&mut img, mx + dx, my + dy, [166, 71, 9]);
                    }
                }
            }
        }
        // 兵刚走到的格子(a3)字形中心叠金色小标记: 均色被拉向炮色, 但深色墨迹仍占多数,
        // 应判回兵而不是丢失
        if board[3][0] == SOLDIER {
            let my = (origin.1 + cell * 3) as i64;
            for dy in -3i64..=3 {
                for dx in -3i64..=3 {
                    put(&mut img, origin.0 as i64 + dx, my + dy, [166, 71, 9]);
                }
            }
        }
        // 棋子: 橙面木盘 + 棕色盘环 + 中心字形(兵黑/炮红)
        for y in 0..game::SIZE {
            for x in 0..game::SIZE {
                let piece = board[y][x];
                if piece == EMPTY {
                    continue;
                }
                let cx = (origin.0 + cell * x as u32) as i64;
                let cy = (origin.1 + cell * y as u32) as i64;
                let r = (cell as f32 * 0.42) as i64;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let d2 = dx * dx + dy * dy;
                        let c = if d2 <= (r - 3) * (r - 3) {
                            [224, 137, 68] // 盘面(两通道都不掩码)
                        } else if d2 <= r * r {
                            [134, 64, 31] // 盘环(两通道都不掩码)
                        } else {
                            continue;
                        };
                        put(&mut img, cx + dx, cy + dy, c);
                    }
                }
                let g = (cell as f32 * 0.16) as i64;
                let glyph: [u8; 3] = if piece == CANNON { [204, 53, 36] } else { [30, 25, 20] };
                for dy in -g..=g {
                    for dx in -g..=g {
                        // 十字形笔画(模拟字形 ~0.3 的填充率, 实心块会被填充率过滤剔除)
                        if dx.abs() <= 1 || dy.abs() <= 1 {
                            put(&mut img, cx + dx, cy + dy, glyph);
                        }
                    }
                }
            }
        }
        // 底部按钮条(全宽暗条, 应被长宽比过滤剔除)
        let by = (origin.1 + cell * 5 + 40) as i64;
        for y in by..by + 36 {
            for x in 10..w as i64 - 10 {
                put(&mut img, x, y, [50, 48, 46]);
            }
        }
        img
    }

    #[test]
    fn test_detect_first_wechat_style() {
        // 微信变体开局(炮居中三列且在下方): 应自动走字形通道识别
        let board = game::middle_startpos_flipped();
        let img = synth_wechat(&board, (120, 100), 40);
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("微信风格识别失败: {e}"));
        assert_eq!(det.board, board);
        assert!(!det.model.cannons_top);
        assert!(det.model.glyph, "橙红底黑红字平台应走字形通道");
        // 中局跟踪: 炮 c4 上移一步后, 兵 a2 下移应手(刚走的兵带金色标记, 不应丢失)
        let mut next = game::apply_move(&board, "c4c3");
        next = game::apply_move(&next, "a2a3");
        let img2 = synth_wechat(&next, (120, 100), 40);
        let crop_img = crop_image(&img2, det.crop);
        let detected = detect_frame(&det.model, &crop_img).unwrap();
        assert_eq!(detected, next, "被标记的兵不应丢失");
    }

    #[test]
    fn test_detect_first_wechat_real_screenshot() {
        // 用户微信小程序窗口的真实首帧截图(橙红底, 黑字兵/红字炮, 炮居中三列在下方)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/first_frame_real.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("微信小程序真实截图识别失败: {e}"));
        assert_eq!(det.board, game::middle_startpos_flipped());
        assert!(!det.model.cannons_top);
        assert!(det.model.glyph, "真实小程序截图应走字形通道");
    }

    #[test]
    fn test_detect_first_midgame_color() {
        // 对局进行中(炮隔空吃兵 + 兵上移)启动监听: 不依赖开局指纹, 应能自举网格与颜色
        let board = game::apply_move(&game::startpos_board(), "a0a2");
        let board = game::apply_move(&board, "b2b1");
        let img = synth(&board, (57, 93), 41, 42, false);
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("任意局面识别失败: {e}"));
        assert_eq!(det.board, board);
        assert!(det.model.cannons_top);
        // 任意局面自举出的模型后续中局跟踪应正常
        let next = game::apply_move(&board, "c0c2");
        let img2 = synth(&next, (57, 93), 41, 7, false);
        let crop_img = crop_image(&img2, det.crop);
        assert_eq!(detect_frame(&det.model, &crop_img).unwrap(), next);
    }

    #[test]
    fn test_detect_first_midgame_wechat() {
        // 微信小程序平台中局启动: 炮 c4 上移(家格露空槽+金色标记)、兵 a2 下移到 a3
        // (字形中心叠金色标记), 横幅/头像/按钮条噪声下应能自举且不丢被标记的兵
        let board = game::apply_move(&game::middle_startpos_flipped(), "c4c3");
        let board = game::apply_move(&board, "a2a3");
        let img = synth_wechat(&board, (120, 100), 40);
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("微信风格任意局面识别失败: {e}"));
        assert_eq!(det.board, board);
        assert!(!det.model.cannons_top, "炮在下方");
        assert!(det.model.glyph, "橙红底平台应走字形通道");
        // 金色标记不应被当作第 4 门炮
        assert_eq!(game::counts(&det.board), (3, 15));
    }

    #[test]
    fn test_detect_first_late_midgame_wechat() {
        // 三炮全部离开家行: 家行整行空出, 网格端点出现平移歧义,
        // 行归属按"空端行在炮侧"(炮在下方 -> 底行是空的家行)推断应得到正确局面
        let board = game::apply_move(&game::middle_startpos_flipped(), "b4b3");
        let board = game::apply_move(&board, "d4d3");
        let board = game::apply_move(&board, "c4c3");
        let img = synth_wechat(&board, (120, 100), 40);
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("微信风格残局识别失败: {e}"));
        assert_eq!(det.board, board);
        assert!(!det.model.cannons_top);
    }

    #[test]
    fn test_model_shifted_keeps_lattice() {
        // 网格平移一格后仍在(平移后的)格点上吸附出整体平移的局面:
        // 这是"纠"按钮平移对齐(worker::align_model)的几何基础
        let board = game::middle_startpos_board();
        let img = synth(&board, (57, 93), 41, 42, false);
        let det = detect_first(&img, None).unwrap();
        let m = det.model.shifted(1, 0);
        let mut expect = [[game::EMPTY; game::SIZE]; game::SIZE];
        for y in 0..game::SIZE {
            for x in 1..game::SIZE {
                expect[y][x - 1] = board[y][x];
            }
        }
        assert_eq!(detect_frame(&m, &crop_image(&img, det.crop)).unwrap(), expect);
    }

    #[test]
    fn test_detect_first_midgame_real_screenshot() {
        // 用户微信小程序窗口的真实中局截图(2026-10-16 "任意局面启动"需求的原始报错现场,
        // 由截图存到 target/midgame_real.png, 不存在时跳过): 炮 b0/d0/a2 在上,
        // 兵 13 枚, 底部有成排按钮噪声, 腐蚀通道字形质心相对格点整体偏移
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/midgame_real.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("真实中局截图识别失败: {e}"));
        let rows = [" C C ", " S   ", "C  SS", "SSSSS", "SSSSS"];
        let mut expect = [[EMPTY; game::SIZE]; game::SIZE];
        for (y, row) in rows.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                expect[y][x] = c;
            }
        }
        assert_eq!(det.board, expect);
        assert!(det.model.cannons_top);
        assert!(det.model.glyph, "真实小程序截图应走字形通道");
    }

    #[test]
    fn test_detect_first_midgame_real2_screenshot() {
        // 用户微信小程序窗口的第二张真实中局截图(2026-10-16 "炮的候选仅 2 个"报错现场):
        // 炮 b0/c0 在上(变体开局位, d2 为前进炮), 兵 14 枚, 第 1 行空。
        // 该帧暴露两个坑: 棋盘下方装饰行与真棋盘在窗口打分上并列(需枚举窗口+
        // 分类校验), 以及并列间距中更大者把格点倾斜到炮的容差外(需 minimax 裁决)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/midgame_real2.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("真实中局截图2识别失败: {e}"));
        let rows = [" CC  ", "     ", "SSSCS", "SSSSS", "SSSSS"];
        let mut expect = [[EMPTY; game::SIZE]; game::SIZE];
        for (y, row) in rows.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                expect[y][x] = c;
            }
        }
        assert_eq!(det.board, expect);
        assert!(det.model.cannons_top);
        assert!(det.model.glyph);
    }

    #[test]
    fn test_detect_first_real3_scaled_window() {
        // 用户把小程序窗口缩小 ~9% 后的截图(2026-10-16 "棋盘横纵间距不一致"报错现场):
        // 格距 100->91.5, 炮字笔画细到腐蚀通道提取不到(只剩 1 个炮块), 且底部
        // 按钮噪声行把 y 轴拟到 145。应通过"可信轴间距约束重拟 + 逐格红像素
        // 密度找炮"自动识别出变体开局(兵 15 三行在上, 炮 b4/c4/d4 在下)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/midgame_real3.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("缩小窗口截图识别失败: {e}"));
        assert_eq!(det.board, game::middle_startpos_flipped());
        assert!(!det.model.cannons_top);
        assert!(det.model.glyph);
        // 中局跟踪走逐帧红像素兜底, 应给出同一局面
        let crop_img = crop_image(&img, det.crop);
        assert_eq!(detect_frame(&det.model, &crop_img).unwrap(), det.board);
    }

    #[test]
    fn test_detect_first_with_color_hint() {
        // "点两下"标注链路: 在真实缩小窗口截图上采样炮/兵颜色作提示,
        // 已知颜色应让识别稳定成功(与自动路径结果一致)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/midgame_real3.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let cannon = sample_camp_color(&img, 339.0, 684.0).expect("炮色采样失败");
        let soldier = sample_camp_color(&img, 79.0, 302.0).expect("兵色采样失败");
        let hint = ColorHint { cannon, soldier };
        let det = detect_first(&img, Some(&hint)).unwrap_or_else(|e| panic!("带提示识别失败: {e}"));
        assert_eq!(det.board, game::middle_startpos_flipped());
    }

    #[test]
    fn test_detect_first_wolf_game_screenshot() {
        // 狼吃羊(同平台另一款游戏, 棋子为红字狼/黑字羊)开局截图: 棋盘同为 5x5,
        // 阵型与变体开局同构(羊三行在上、隔一空行、狼居中三列在底)。
        // 现有管线应零改动识别成功(狼->炮 羊->兵 映射) -- 狼吃羊支持只差
        // 斜线区走法等规则差异
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/wolf_real.png");
        let Ok(reader) = xcap::image::ImageReader::open(path) else {
            return;
        };
        let img = reader.decode().expect("解码截图失败").to_rgba8();
        let det = detect_first(&img, None).unwrap_or_else(|e| panic!("狼吃羊截图识别失败: {e}"));
        assert_eq!(det.board, game::middle_startpos_flipped());
        assert!(!det.model.cannons_top);
        assert!(det.model.glyph);
    }

    #[test]
    fn test_pos_roundtrip() {
        for y in 0..game::SIZE {
            for x in 0..game::SIZE {
                let pos = pos_of(x, y);
                assert_eq!(xy_of(&pos), Some((x, y)));
            }
        }
    }
}
