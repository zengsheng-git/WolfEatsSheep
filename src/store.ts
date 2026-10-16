import { reactive } from "vue";

import type { ArrowHint } from "./arrows";

// 连线分析的实时状态
export const mirrorStore = reactive({
    board: {} as Record<string, string>, // 当前局面(坐标 -> 棋子, ' ' 为空)
    camp: "w", // 当前行棋方 'w'炮方 / 'b'兵方
    mirror: false, // 是否镜像视角(把我方翻到下方显示)
    myCamp: "w", // 我方阵营 'w'炮 / 'b'兵, 启动监听后由后端按画面自动判定写入
    arrows: [] as ArrowHint[], // 招法提示箭头(最优/次优), 分析面板写入
    calibrating: false, // 纠错模式: 棋盘可点击标注
    editBoard: {} as Record<string, string>, // 纠错模式下的标注局面(逻辑坐标)
    calibTurn: "w" as "w" | "b", // 标注后的行棋方(对应平台"当前走棋"指示)
});
