// 三炮十五兵棋盘识别: 经典 CV 管线, 替代原项目的 YOLO ONNX 模型
// 思路: 棋子是棋盘上唯一的"圆形色块" -> 连通域提取 -> 由开局阵型指纹拟合 5x5 网格
//       -> 开局位置自举两方颜色质心 -> 后续帧按网格归属 + 最近色心分类
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

// 首帧识别: 依次尝试多个通道, 任一通道拟合出开局指纹即成功
// 通道 = (是否字形掩码, 腐蚀半径):
//   实心彩色棋子平台走彩色通道; 橙红底+黑红字平台(微信小程序)走字形通道;
//   腐蚀变体切断被掩码的网格线, 避免线条把棋子连成片
pub fn detect_first(img: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<FirstDetect, String> {
    let (iw, ih) = img.dimensions();
    let channels: [(bool, usize); 4] = [(false, 0), (false, 2), (true, 0), (true, 2)];
    // 全部失败时返回"走得最远"(色块最多)的通道的错误, 对用户信息量最大
    let mut best_err: Option<(String, usize)> = None;
    for &(glyph, erode) in &channels {
        let (blobs, step) = extract_blobs(img, 1024, erode, glyph);
        match fit_startpos(&blobs, step, iw, ih, glyph, erode != 0) {
            Ok(det) => return Ok(det),
            Err(e) => {
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
    if cannon_cands.len() < 3 {
        return None;
    }
    cannon_cands.sort_by(|a, b| a.2.total_cmp(&b.2));
    let cannon_pts: Vec<(usize, usize)> = cannon_cands.iter().take(3).map(|&(x, y, _)| (x, y)).collect();

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


// 装饰块过滤(字形通道): 棋子字形尺寸一致, 而炮位空槽等装饰是无彩深色且明显更大
fn make_deco_filter(glyph: bool, blobs: &[Blob]) -> impl Fn(&Blob) -> bool + '_ {
    let mut areas: Vec<usize> = blobs.iter().map(|b| b.area).collect();
    areas.sort_unstable();
    let med = areas.get(areas.len() / 2).map(|v| *v as f32).unwrap_or(f32::MAX);
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
        let det = detect_first(&img).unwrap();
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
        let det = detect_first(&img).unwrap();
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
        let det = detect_first(&img).unwrap();
        assert_eq!(det.board, flipped_startpos());
        assert!(!det.model.cannons_top);
    }

    #[test]
    fn test_detect_frame_midgame() {
        let board = game::startpos_board();
        let img = synth(&board, (57, 93), 41, 42, false);
        let det = detect_first(&img).unwrap();
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
        let det = detect_first(&img).unwrap();
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
        let det = detect_first(&img).unwrap_or_else(|e| panic!("真实浏览器截图识别失败: {e}"));
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
        let det = detect_first(&img1).unwrap_or_else(|e| panic!("开局截图识别失败: {e}"));
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
        // 应被"炮恒 3"约束当作标记丢弃
        if cannon_rows.len() != 1 && board[4][2] == EMPTY {
            let mx = (origin.0 + cell * 2) as i64;
            let my = (origin.1 + cell * 4) as i64;
            for dy in -6i64..=6 {
                for dx in -6i64..=6 {
                    put(&mut img, mx + dx, my + dy, [166, 71, 9]);
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
        let det = detect_first(&img).unwrap_or_else(|e| panic!("微信风格识别失败: {e}"));
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
        let det = detect_first(&img).unwrap_or_else(|e| panic!("微信小程序真实截图识别失败: {e}"));
        assert_eq!(det.board, game::middle_startpos_flipped());
        assert!(!det.model.cannons_top);
        assert!(det.model.glyph, "真实小程序截图应走字形通道");
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
