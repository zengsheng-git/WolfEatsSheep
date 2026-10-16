<script setup lang="ts">
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { onMounted, ref, h, computed } from "vue";
import { useDialog, useMessage } from "naive-ui";
import {
    NButton,
    NCard,
    NFlex,
    NForm,
    NFormItem,
    NInputNumber,
    NSelect,
    NDrawer,
    NDrawerContent,
    NSpace,
    NTooltip,
    NEmpty,
    NScrollbar,
    NTag,
    NInput,
} from "naive-ui";
import { mirrorStore } from "../store";

interface EngineConfig {
    depth: number;
    time: number;
    alt_score_gap: number;
    hint_level: number;
    my_camp: string;
}

// 缩放由 App 应用到窗口
const zoom = defineModel<number>("zoom", { default: 1 });

const zoomOptions = [
    { label: "60%", value: 0.6 },
    { label: "70%", value: 0.7 },
    { label: "80%", value: 0.8 },
    { label: "90%", value: 0.9 },
    { label: "100%", value: 1 },
    { label: "110%", value: 1.1 },
    { label: "125%", value: 1.25 },
];

// 提示强度档位映射到固定搜索深度: 每一手都出自同样的浅思考, 整体一致地弱
const hintOptions = [
    { label: "最强(不限)", value: 0 },
    { label: "新手(深度2)", value: 1 },
    { label: "入门(深度3)", value: 2 },
    { label: "初级(深度4)", value: 3 },
    { label: "中级(深度5)", value: 4 },
    { label: "高级(深度6)", value: 5 },
];

const campOptions = [
    { label: "执炮(先行)", value: "w" },
    { label: "执兵(后行)", value: "b" },
];

const config = ref<EngineConfig>({
    depth: 6,
    time: 1,
    alt_score_gap: 100,
    hint_level: 0,
    my_camp: "w",
});

const showEngineConfig = ref(false);
const isEngineRunning = ref(false);

const dialog = useDialog();
const message = useMessage();

onMounted(async () => {
    await getEngineConfig();
    // 目标窗口关闭等异常导致监听结束时, 同步按钮状态并提示
    listen("listen_stopped", (event) => {
        isEngineRunning.value = false;
        dialog.warning({
            title: "监听停止",
            content: String(event.payload),
            positiveText: "确定",
            showIcon: true,
        });
    });
});

async function stopListen() {
    await invoke("stop_listen");
    isEngineRunning.value = false;
}

interface Window {
    id: number;
    title: string;
    app_name: string;
    width: number;
    height: number;
}

async function startListen() {
    try {
        // 获取窗口列表
        const windows: Window[] = await invoke("list_windows");

        if (windows.length === 0) {
            dialog.warning({
                title: "警告",
                content: "没有找到可用的窗口",
                positiveText: "确定",
                showIcon: true,
            });
            return;
        }

        const selectedWindowId = ref<number | null>(null);
        const searchQuery = ref("");

        const filteredWindows = computed(() => {
            if (!searchQuery.value) return windows;
            const query = searchQuery.value.toLowerCase();
            return windows.filter(
                (w) => w.title.toLowerCase().includes(query) || w.app_name.toLowerCase().includes(query)
            );
        });

        // 显示窗口选择对话框
        dialog.info({
            title: "选择要监听的窗口(需显示标准开局棋盘)",
            class: "window-select-dialog",
            content: () =>
                h(NFlex, { vertical: true, style: "gap: 16px" }, [
                    h(NInput, {
                        clearable: true,
                        placeholder: "搜索窗口...",
                        "onUpdate:value": (val) => (searchQuery.value = val),
                        style: "width: 100%",
                    }),
                    h(NScrollbar, { style: "max-height: 300px" }, [
                        filteredWindows.value.length > 0
                            ? h(
                                  NSpace,
                                  { vertical: true, size: "small" },
                                  filteredWindows.value.map((w) =>
                                      h(
                                          NCard,
                                          {
                                              hoverable: true,
                                              size: "small",
                                              bordered: true,
                                              class: selectedWindowId.value === w.id ? "selected-window" : "",
                                              onClick: () => (selectedWindowId.value = w.id),
                                          },
                                          {
                                              default: () => [
                                                  h(NFlex, { align: "center", justify: "space-between" }, [
                                                      h("div", [
                                                          h("div", { class: "window-title" }, w.title),
                                                          h("div", { class: "window-app" }, [
                                                              w.app_name,
                                                              h(
                                                                  NTag,
                                                                  {
                                                                      size: "tiny",
                                                                      type: "info",
                                                                      style: "margin-left: 8px",
                                                                  },
                                                                  { default: () => `${w.width}×${w.height}` }
                                                              ),
                                                          ]),
                                                      ]),
                                                      h(
                                                          NButton,
                                                          {
                                                              tertiary: true,
                                                              circle: true,
                                                              type:
                                                                  selectedWindowId.value === w.id
                                                                      ? "primary"
                                                                      : "default",
                                                              size: "small",
                                                          },
                                                          {
                                                              default: () =>
                                                                  selectedWindowId.value === w.id ? "✓" : "",
                                                          }
                                                      ),
                                                  ]),
                                              ],
                                          }
                                      )
                                  )
                              )
                            : h(NEmpty, { description: "没有找到匹配的窗口" }),
                    ]),
                ]),
            positiveText: "确定",
            negativeText: "取消",
            style: "max-width: 500px",
            maskClosable: false,
            onPositiveClick: async () => {
                if (!selectedWindowId.value) {
                    dialog.warning({
                        title: "提示",
                        content: "请先选择一个窗口",
                        positiveText: "确定",
                    });
                    return false; // 阻止对话框关闭
                }

                const window = windows.find((w) => w.id === selectedWindowId.value);
                if (window) {
                    try {
                        await invoke("start_listen", { target: window });
                        isEngineRunning.value = true;
                    } catch (error) {
                        dialog.error({
                            title: "启动监听失败",
                            content: String(error),
                            positiveText: "确定",
                        });
                    }
                }
            },
        });
    } catch (error) {
        console.error("启动监听失败:", error);
        dialog.error({
            title: "错误",
            content: "启动监听失败: " + String(error),
            positiveText: "确定",
        });
    }
}

async function setEngineDepth() {
    await invoke("set_engine_depth", { depth: config.value.depth });
}

async function setEngineTime() {
    // 界面单位为秒, 后端保存时转换为毫秒
    await invoke("set_engine_time", { time: config.value.time });
}

async function setEngineAltScoreGap() {
    await invoke("set_engine_alt_score_gap", { gap: config.value.alt_score_gap });
}

async function setEngineHintLevel() {
    await invoke("set_engine_hint_level", { level: config.value.hint_level });
}

async function setMyCamp() {
    await invoke("set_my_camp", { camp: config.value.my_camp });
    mirrorStore.myCamp = config.value.my_camp;
}

async function getEngineConfig() {
    const result: EngineConfig = await invoke("get_engine_config");
    config.value = {
        ...result,
        time: Number((result.time / 1000).toFixed(1)),
    };
    mirrorStore.myCamp = result.my_camp === "b" ? "b" : "w";
}

async function toggleEngine() {
    isEngineRunning.value ? await stopListen() : await startListen();
}

// ---- 手动纠错: 以镜像棋盘的标注局面重校准颜色与同步状态 ----

function startCalibrate() {
    // 以当前显示局面为基准开始标注; 行棋方默认取当前值, 与平台"当前走棋"指示核对后可改
    mirrorStore.editBoard = { ...mirrorStore.board };
    mirrorStore.calibTurn = mirrorStore.camp === "b" ? "b" : "w";
    mirrorStore.calibrating = true;
}

function cancelCalibrate() {
    mirrorStore.calibrating = false;
}

// 重新识别: 按当前已知局面重学炮/兵颜色特征(棋盘显示正确但识别走样时用)
async function recalibrate() {
    try {
        await invoke("recalibrate");
        message.success("已按当前局面重新识别炮/兵特征");
    } catch (error) {
        dialog.error({
            title: "重新识别失败",
            content: String(error),
            positiveText: "确定",
        });
    }
}

async function applyCalibrate() {
    // 序列化为 5 行字符串('C'炮/'S'兵/' '空)
    const rows: string[] = [];
    for (let y = 0; y < 5; y++) {
        let row = "";
        for (let x = 0; x < 5; x++) {
            row += mirrorStore.editBoard[`${"abcde"[x]}${y}`] ?? " ";
        }
        rows.push(row);
    }
    try {
        await invoke("calibrate", { board: rows, turn: mirrorStore.calibTurn });
        mirrorStore.calibrating = false;
    } catch (error) {
        dialog.error({
            title: "纠错失败",
            content: String(error),
            positiveText: "确定",
        });
    }
}
</script>

<template>
    <n-card class="toolbar" :bordered="false" size="small">
        <n-space vertical size="small">
            <n-flex align="center" justify="space-between">
                <n-flex align="center" :size="8">
                    <n-select v-model:value="config.my_camp" :options="campOptions" :consistent-menu-width="false"
                        placeholder="我方阵营" class="camp-select" @update:value="setMyCamp" />
                    <n-tooltip trigger="hover" placement="bottom">
                        <template #trigger>
                            <n-select
                                size="small"
                                v-model:value="zoom"
                                :options="zoomOptions"
                                :consistent-menu-width="false"
                                class="zoom-select"
                            />
                        </template>
                        界面缩放
                    </n-tooltip>
                </n-flex>

                <n-space>
                    <template v-if="mirrorStore.calibrating">
                        <n-tooltip trigger="hover" placement="bottom">
                            <template #trigger>
                                <n-button size="tiny" :type="mirrorStore.calibTurn === 'w' ? 'primary' : 'default'"
                                    @click="mirrorStore.calibTurn = 'w'">
                                    轮到炮
                                </n-button>
                            </template>
                            对应小程序里"当前走棋"的指示
                        </n-tooltip>
                        <n-button size="tiny" :type="mirrorStore.calibTurn === 'b' ? 'primary' : 'default'"
                            @click="mirrorStore.calibTurn = 'b'">
                            轮到兵
                        </n-button>
                        <n-tooltip trigger="hover" placement="bottom">
                            <template #trigger>
                                <n-button circle size="small" type="primary" @click="applyCalibrate">✓</n-button>
                            </template>
                            应用标注并重校准识别
                        </n-tooltip>
                        <n-tooltip trigger="hover" placement="bottom">
                            <template #trigger>
                                <n-button circle size="small" @click="cancelCalibrate">✗</n-button>
                            </template>
                            放弃标注
                        </n-tooltip>
                    </template>

                    <template v-else>
                        <n-tooltip trigger="hover" placement="bottom">
                            <template #trigger>
                                <n-button
                                    circle
                                    size="small"
                                    :type="isEngineRunning ? 'error' : 'primary'"
                                    @click="toggleEngine"
                                >
                                    {{ isEngineRunning ? "停" : "启" }}
                                </n-button>
                            </template>
                            {{ isEngineRunning ? "停止监听" : "开始监听" }}
                        </n-tooltip>

                        <n-tooltip trigger="hover" placement="bottom" v-if="isEngineRunning">
                            <template #trigger>
                                <n-button circle size="small" type="warning" @click="startCalibrate">纠</n-button>
                            </template>
                            手动纠错: 标注识别错误的棋子, 重新校准识别
                        </n-tooltip>

                        <n-tooltip trigger="hover" placement="bottom" v-if="isEngineRunning">
                            <template #trigger>
                                <n-button circle size="small" type="success" @click="recalibrate">识</n-button>
                            </template>
                            重新识别: 按当前局面重学炮/兵颜色特征(棋盘显示正确但识别走样时用)
                        </n-tooltip>

                        <n-tooltip trigger="hover" placement="bottom">
                            <template #trigger>
                                <n-button circle size="small" type="info" @click="showEngineConfig = true">配</n-button>
                            </template>
                            引擎配置
                        </n-tooltip>
                    </template>
                </n-space>
            </n-flex>
        </n-space>

        <n-drawer v-model:show="showEngineConfig" :width="300" placement="right">
            <n-drawer-content title="引擎配置">
                <n-form :model="config" label-placement="left" label-width="80">
                    <n-form-item label="提示强度">
                        <n-select
                            v-model:value="config.hint_level"
                            :options="hintOptions"
                            :consistent-menu-width="false"
                            style="width: 220px"
                            @update:value="setEngineHintLevel"
                        />
                    </n-form-item>
                    <n-form-item label="深度">
                        <n-input-number
                            v-model:value="config.depth"
                            button-placement="both"
                            :min="1"
                            :max="16"
                            style="width: 120px"
                            @update:value="setEngineDepth"
                        />
                    </n-form-item>
                    <n-form-item label="时间(秒)">
                        <n-input-number
                            v-model:value="config.time"
                            button-placement="both"
                            :step="0.5"
                            :precision="1"
                            :min="0.2"
                            :max="30"
                            style="width: 120px"
                            @update:value="setEngineTime"
                        />
                    </n-form-item>
                    <n-form-item label="次优分差">
                        <n-input-number
                            v-model:value="config.alt_score_gap"
                            button-placement="both"
                            :min="0"
                            :max="1000"
                            :step="10"
                            style="width: 120px"
                            @update:value="setEngineAltScoreGap"
                        />
                    </n-form-item>
                </n-form>
            </n-drawer-content>
        </n-drawer>
    </n-card>
</template>

<style scoped>
.toolbar {
    width: 100%;
    padding: 8px;
    border-radius: 8px;
}

.camp-select {
    width: 120px;
}

.zoom-select {
    width: 88px;
}

:deep(.n-button) {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
    transition: all 0.3s;
}

:deep(.n-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

:deep(.window-title) {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 360px;
}

:deep(.window-app) {
    font-size: 12px;
    color: #999;
    margin-top: 4px;
    display: flex;
    align-items: center;
}

:deep(.selected-window) {
    border-color: var(--primary-color) !important;
    background-color: rgba(var(--primary-color-rgb), 0.05);
}
</style>
