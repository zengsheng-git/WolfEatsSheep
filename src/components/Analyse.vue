<script setup lang="ts">
import { listen } from '@tauri-apps/api/event';
import { NCard, NFlex, NTag, NText } from 'naive-ui';
import { computed, ref } from 'vue';

import type { ArrowHint } from '../arrows';
import { evalText, formatGap } from '../format';
import { mirrorStore } from '../store';

interface Analyse {
    depth: number,   // 深度
    score: number,   // 得分(行棋方视角)
    time: number,    // 时间(毫秒)
    pvs: string[],   // 思考主线(iccs)
    moves: string[], // 主线中文记谱
    alternatives: string[], // 次优首着(iccs)
    alt_scores: number[], // 次优与最优分差, 负值=未知
    alt_moves: string[], // 次优中文记谱
    deviation: { camp: string; loss: number } | null, // 上一步非预期走子的代价
    board: { piece: string; pos: string }[], // 分析时局面, 供主线预演
    state: string,   // 状态
    source: string,  // 来源
    camp: string,    // 行棋方阵营 'w'炮 / 'b'兵
    over: string | null, // 终局提示
}

const followUps = ref<string[]>([])
const pvs = ref<string[]>([])
const altIccs = ref<string[]>([])
const deviation = ref<{ camp: string; loss: number } | null>(null)
const lastScore = ref(0)
const analysisBoard = ref<{ piece: string; pos: string }[]>([])
const alternatives = ref<string[]>([])
const altScores = ref<number[]>([])
const over = ref<string | null>(null)
// 最优招法的行棋方阵营('w'炮/'b'兵), 后续招法按行棋方交替推演
const bestCamp = ref<"w" | "b">("w")
const best = ref({
    move: "----",
    side: "--",
    depth: 0,
    source: "--",
    evalText: "--",
    evalType: "info" as "error" | "success" | "info",
})

listen('analyse', async (event) => {
    let data = event.payload as Analyse;
    best.value.move = data.moves[0] ?? "----";
    best.value.depth = data.depth;
    best.value.source = data.source;
    lastScore.value = data.score;
    pvs.value = data.pvs ?? [];
    altIccs.value = data.alternatives ?? [];
    deviation.value = data.deviation ?? null;
    analysisBoard.value = data.board ?? [];
    over.value = data.over ?? null;
    // 新分析到达时旧主线已过期, 若正处预演态则强制还原, 避免悬停元素被重渲染后 mouseleave 丢失导致预演卡死
    if (previewActive) restoreBoard();
    alternatives.value = data.alt_moves ?? [];
    altScores.value = data.alt_scores ?? [];
    followUps.value = data.moves;
    const isMine = data.camp === mirrorStore.myCamp;
    bestCamp.value = data.camp === "b" ? "b" : "w";
    best.value.side = data.camp === "w" ? "炮方" : "兵方";
    const evalResult = formatEval(data.score, isMine);
    best.value.evalText = evalResult.text;
    best.value.evalType = evalResult.type;

    // 招法提示箭头: 最优招法按行棋方着色(我方红/对方蓝), 次优为紫色细箭头
    const arrows: ArrowHint[] = [];
    const pv = data.pvs[0];
    if (pv && pv.length >= 4) {
        arrows.push({ from: pv.substring(0, 2), to: pv.substring(2, 4), kind: isMine ? "red" : "blue" });
    }
    for (const alt of data.alternatives ?? []) {
        if (alt.length >= 4) {
            arrows.push({ from: alt.substring(0, 2), to: alt.substring(2, 4), kind: "alt" });
        }
    }
    mirrorStore.arrows = arrows;
})

function campName(camp: string): string {
    return camp === "w" ? "炮方" : "兵方";
}

// 分数解读：行棋方视角文字, 颜色按我方利益着色：红=利好我方, 绿=利好对方
function formatEval(score: number, isMine: boolean): { text: string, type: "error" | "success" | "info" } {
    const goodForMe = isMine === (score > 0);
    return { text: evalText(score), type: goodForMe ? "error" : "success" };
}

// 主线着法按行棋方交替: 偶数索引是最优招法的行棋方, 奇数索引是对手
function followUpSide(index: number): string {
    const moverCannon = bestCamp.value === "w";
    const isMover = index % 2 === 0;
    return (isMover ? moverCannon : !moverCannon) ? "炮方" : "兵方";
}

// 偏离提示代价: 颜色沿用面板约定, 红=利好我方, 绿=利好对方
const deviationText = computed(() => {
    if (!deviation.value) return null;
    const mine = deviation.value.camp === mirrorStore.myCamp;
    const who = campName(deviation.value.camp);
    const loss = deviation.value.loss;
    if (Math.abs(loss) <= 30) {
        return { text: `${who}偏离提示, 与预期相当`, type: "info" as const };
    }
    const goodForMe = mine ? loss < 0 : loss > 0;
    const type = goodForMe ? ("error" as const) : ("success" as const);
    return loss > 0
        ? { text: `${who}偏离提示, 亏 ${loss} 分`, type }
        : { text: `${who}偏离提示, 反赚 ${-loss} 分`, type };
});

// ===== 主线预演: 通知棋盘组件从分析时局面重放着法, 移开按最新实时局面还原 =====

let previewActive = false;

// 悬停主线第 index 项: 预演到该步的局面(含最优招法)
function previewLine(index: number) {
    if (!pvs.value.length || !analysisBoard.value.length) return;
    previewActive = true;
    window.dispatchEvent(new CustomEvent("preview-line", {
        detail: { board: analysisBoard.value, pvs: pvs.value, count: Math.min(index + 1, pvs.value.length) },
    }));
}

function restoreBoard() {
    if (!previewActive) return;
    previewActive = false;
    window.dispatchEvent(new CustomEvent("preview-end"));
}

// 悬停次优招法: 预演走该备选首着后的局面
function previewAlt(index: number) {
    const alt = altIccs.value[index];
    if (!alt || alt.length < 4 || !analysisBoard.value.length) return;
    previewActive = true;
    window.dispatchEvent(new CustomEvent("preview-line", {
        detail: { board: analysisBoard.value, pvs: [alt], count: 1 },
    }));
}

</script>

<template>
    <n-card title="局面分析" :bordered="false" class="textlog">
        <n-flex justify="space-between" align="end">
            <n-text type="info" class="analyse-title" strong @mouseenter="previewLine(0)" @mouseleave="restoreBoard()">
                {{ best.side }} {{ best.move }}
            </n-text>
            <n-flex align="center" :size="6">
                <n-tag size="small" round :bordered="false" type="info">{{ best.source }}</n-tag>
                <n-text type="warning">深度 {{ best.depth }}</n-text>
            </n-flex>
        </n-flex>
        <n-flex>
            <n-text :type="best.evalType">
                {{ best.evalText }}
            </n-text>
        </n-flex>
        <n-tag v-if="over" type="warning" size="small" round :bordered="false" style="margin-top: 6px">{{ over }}</n-tag>
        <n-text v-if="deviationText" :type="deviationText.type" class="notes">{{ deviationText.text }}</n-text>
        <div v-if="followUps.length" class="follow-ups">
            <n-text depth="3">后续：</n-text>
            <div
                v-for="(m, i) in followUps"
                :key="i"
                class="follow-up-item"
                @mouseenter="previewLine(i)"
                @mouseleave="restoreBoard()"
            >
                <n-text depth="3">{{ i + 1 }}. {{ followUpSide(i) }} {{ m }}</n-text>
            </div>
        </div>
        <div v-if="alternatives.length" class="alternatives">
            <n-text strong style="color: #9b59b6">次优招法</n-text>
                    <div
                        v-for="(m, i) in alternatives"
                        :key="i"
                        class="alt-item"
                        @mouseenter="previewAlt(i)"
                        @mouseleave="restoreBoard()"
                    >
                        <n-text depth="3">{{ i + 2 }}. {{ m }} {{ formatGap(altScores[i]) }}</n-text>
                    </div>
        </div>
    </n-card>
</template>

<style scoped>
.notes {
    margin-top: 6px;
}

.follow-ups {
    margin-top: 6px;
    max-height: 260px;
    overflow-y: auto;
    font-size: 12px;
}

.follow-up-item {
    line-height: 1.7;
    border-radius: 2px;
    padding: 0 2px;
}

.follow-up-item:hover {
    background-color: rgba(24, 160, 88, 0.15);
}

.analyse-title {
    font-size: x-large;
    cursor: pointer;
    border-radius: 2px;
    padding: 0 2px;
}

.analyse-title:hover {
    background-color: rgba(24, 160, 88, 0.15);
}

.alternatives {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.alt-item {
    line-height: 1.5;
    border-radius: 2px;
    padding: 0 2px;
    cursor: pointer;
}

.alt-item:hover {
    background-color: rgba(24, 160, 88, 0.15);
}

.textlog {
    width: 300px;
    height: 440px;
    left: 336px;
    top: 104px;
}
</style>
