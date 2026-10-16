<script setup lang="ts">

import { listen } from "@tauri-apps/api/event";
import { computed, onMounted, ref, watch } from "vue";

import { computeArrows } from "../arrows";
import { mirrorStore } from "../store";
import "../assets/css/chessboard.css";


interface Position {
    piece: string,
    pos: string,
}

interface Changed {
    piece: string,
    from: string,
    to: string,
    camp: string,
}


// 标准开局: 炮 a0/c0/e0, 兵填满 2~4 排
const startpos: Position[] = [
    { piece: "C", pos: "a0" },
    { piece: "C", pos: "c0" },
    { piece: "C", pos: "e0" },
    ...["a", "b", "c", "d", "e"].flatMap((file) =>
        [2, 3, 4].map((rank) => ({ piece: "S", pos: `${file}${rank}` }))
    ),
];

// 棋盘数据态(坐标 -> 棋子), 作为展示与预演的共同来源, DOM 仅按它渲染
const boardMap = ref<Record<string, string>>({});
let previewing = false;

// 后端 y=0 为截屏顶部; 非镜像显示按原样从上到下排列, 镜像时反转
const ALL_IDS: string[] = [];
for (let y = 0; y <= 4; y++) {
    for (let x = 0; x < 5; x++) {
        ALL_IDS.push(`${"abcde"[x]}${y}`);
    }
}

// 招法提示箭头(最优/次优), 由分析面板写入 store, 视角翻转时随镜像重排自动重算; 纠错模式下隐藏
const arrowShapes = computed(() =>
    mirrorStore.calibrating ? [] : computeArrows(mirrorStore.arrows, mirrorStore.mirror)
);

function clearHighlights() {
    // 走子/局面重同步后旧提示箭头已失效, 直接清空
    mirrorStore.arrows = [];
}

function renderBoard(map: Record<string, string>) {
    for (const [pos, piece] of Object.entries(map)) {
        const ele = document.getElementById("chessboard")?.querySelector(`[id="${pos}"]`)?.firstElementChild;
        if (!ele) continue;
        ele.classList.forEach(cls => {
            if (cls !== "piece") {
                ele.classList.remove(cls)
            }
        });
        if (piece !== " ") {
            ele.classList.add(`piece-${piece}`);
        }
    }
}

function applyPosition(pieces: Position[]) {
    const map: Record<string, string> = {};
    for (const id of ALL_IDS) map[id] = " ";
    for (const record of pieces) map[record.pos] = record.piece;
    boardMap.value = map;
    mirrorStore.board = map;
    if (!previewing && !mirrorStore.calibrating) renderBoard(map);
}

function applyMove(change: Changed) {
    const map = { ...boardMap.value };
    map[change.from] = " ";
    map[change.to] = change.piece;
    boardMap.value = map;
    mirrorStore.board = map;
    if (!previewing && !mirrorStore.calibrating) renderBoard(map);
}

// 纠错模式: 点击格子循环切换 空->炮->兵(直接修改标注副本并渲染)
function onCellClick(pos: string) {
    if (!mirrorStore.calibrating) return;
    const cur = mirrorStore.editBoard[pos] ?? " ";
    mirrorStore.editBoard[pos] = cur === " " ? "C" : cur === "C" ? "S" : " ";
    renderBoard(mirrorStore.editBoard);
}

// 退出纠错模式时按最新数据态还原显示
watch(() => mirrorStore.calibrating, (active) => {
    if (!active) renderBoard(boardMap.value);
});

// 主线预演: 从分析时局面的数据副本上重放着法; 预演期间实时 position/move 只更新数据不刷新 DOM
window.addEventListener("preview-line", (event) => {
    if (mirrorStore.calibrating) return;
    const detail = (event as CustomEvent).detail as { board: Position[]; pvs: string[]; count: number };
    if (!detail?.board?.length) return;
    const map: Record<string, string> = {};
    for (const id of ALL_IDS) map[id] = " ";
    for (const record of detail.board) map[record.pos] = record.piece;
    for (let i = 0; i < detail.count && i < detail.pvs.length; i++) {
        const from = detail.pvs[i].substring(0, 2);
        const to = detail.pvs[i].substring(2, 4);
        const piece = map[from];
        if (!piece || piece === " ") break;
        map[from] = " ";
        map[to] = piece;
    }
    previewing = true;
    renderBoard(map);
});

window.addEventListener("preview-end", () => {
    // 按最新数据态还原(预演期间可能已有实时走子)
    previewing = false;
    renderBoard(boardMap.value);
});

onMounted(() => {
    // 初始展示标准开局
    applyPosition(startpos);
});

const wrappedItems = computed(() => {
    const items = [...ALL_IDS];
    return mirrorStore.mirror ? items.reverse() : items;
});

listen('mirror', async (event) => {
    mirrorStore.mirror = event.payload as boolean;
})

listen('position', async (event) => {
    let pos = event.payload as Position[];
    clearHighlights();
    applyPosition(pos);
    // 全量重同步意味着预演已过期, 强制退出预演态
    if (previewing) {
        previewing = false;
        renderBoard(boardMap.value);
    }
})

listen('move', async (event) => {
    let change = event.payload as Changed;
    clearHighlights();
    applyMove(change);
    // 走子即换行棋方; 后端 camp 为刚走子一方("Cannon"/"Soldier")
    mirrorStore.camp = change.camp === "Cannon" ? "b" : "w";
});

// 分析事件携带权威局面, 顺带校正可能积累的显示偏差(预演中仅更新数据, 不打断预演)
listen('analyse', async (event) => {
    const data = event.payload as { board?: Position[]; camp?: string };
    if (!data?.board?.length) return;
    mirrorStore.camp = data.camp ?? mirrorStore.camp;
    applyPosition(data.board);
});

</script>

<template>
    <div id="chessboard" :class="{ calibrating: mirrorStore.calibrating }">
        <svg class="grid-lines" viewBox="0 0 308 308">
            <!-- 5 条横线 + 5 条竖线画在交叉点上, 端点在首末交叉点处 -->
            <line v-for="i in 5" :key="'h' + i" :x1="42" :y1="14 + 28 + (i - 1) * 56" :x2="266"
                :y2="14 + 28 + (i - 1) * 56" stroke="#8a5a2b" stroke-width="2.5" />
            <line v-for="i in 5" :key="'v' + i" :x1="14 + 28 + (i - 1) * 56" :y1="42" :x2="14 + 28 + (i - 1) * 56"
                :y2="266" stroke="#8a5a2b" stroke-width="2.5" />
            <rect x="42" y="42" width="224" height="224" fill="none" stroke="#8a5a2b" stroke-width="3" />
        </svg>
        <div v-for="item in wrappedItems" :key="item" :id="item" class="piece-wrap" @click="onCellClick(item)"><span
                class="piece"></span></div>
        <svg class="hint-arrows" viewBox="0 0 308 308">
            <!-- 先画全部箭杆再画全部箭头, 箭头交叉时保持头部在最上层 -->
            <line v-for="(a, i) in arrowShapes" :key="i" :x1="a.x1" :y1="a.y1" :x2="a.x2" :y2="a.y2" :stroke="a.color"
                :stroke-width="a.width" stroke-linecap="round" />
            <polygon v-for="(a, i) in arrowShapes" :key="'h' + i" :points="a.head" :fill="a.color" />
        </svg>
        <div v-if="mirrorStore.calibrating" class="calib-banner">纠错模式: 点击棋子切换 空→炮→兵</div>
    </div>
</template>

<style scoped>
#chessboard {
    position: relative;
}

#chessboard.calibrating .piece-wrap {
    cursor: pointer;
}

.calib-banner {
    position: absolute;
    top: -6px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(208, 100, 20, 0.92);
    color: #fff;
    font-size: 12px;
    padding: 2px 10px;
    border-radius: 8px;
    z-index: 5;
    white-space: nowrap;
}
</style>
