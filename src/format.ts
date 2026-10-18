// 评分文字解读: 行棋方视角, 正=行棋方占优
export function evalText(score: number): string {
    const abs = Math.abs(score);
    if (score >= 30000) return "必胜"; // 求解器: 已判胜但主线未推到终局, 步数未知
    if (score <= -30000) return "必负";
    if (score >= 29000) return `${30000 - score}步胜`;
    if (score <= -29000) return `${30000 + score}步负`;
    if (abs < 30) return "均势";
    const gradePos = ["略优", "较优", "大优", "胜势"];
    const gradeNeg = ["略差", "较差", "大差", "败势"];
    const idx = abs < 150 ? 0 : abs < 400 ? 1 : abs < 800 ? 2 : 3;
    return score > 0 ? `+${abs} ${gradePos[idx]}` : `-${abs} ${gradeNeg[idx]}`;
}

// 次优招法分差展示: 0 表示与最优同级; 负值=未知, 不展示
export function formatGap(gap: number | undefined): string {
    if (gap === undefined || gap < 0) return "";
    if (gap === 0) return "±0";
    return `-${gap}`;
}
