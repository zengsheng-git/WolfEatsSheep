<script setup lang="ts">
import { getCurrentWindow, LogicalSize } from "@tauri-apps/api/window";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { onMounted, ref, watch } from "vue";
import { NDivider, NLayoutFooter, NDialogProvider, NMessageProvider } from "naive-ui";
import Analyse from "./components/Analyse.vue";
import Chessboard from "./components/Chessboard.vue";
import Toolbar from "./components/Toolbar.vue";

// 界面缩放倍率, 持久化在本地
const zoom = ref(Number(localStorage.getItem("sanpao_zoom")) || 1);

// 按缩放倍率调整 WebView 缩放和窗口尺寸, 保证内容完整显示
async function applyLayout(z: number) {
    try {
        await getCurrentWebview().setZoom(z);
        await getCurrentWindow().setSize(new LogicalSize(Math.round(660 * z), Math.round(580 * z)));
    } catch (e) {
        console.error("调整窗口尺寸失败:", e);
    }
}

watch(zoom, (z) => applyLayout(z));

// 恢复上次的缩放设置并持久化
onMounted(() => {
    localStorage.setItem("sanpao_zoom", String(zoom.value));
    applyLayout(zoom.value);
});
</script>

<template>
    <n-dialog-provider>
        <n-message-provider>
            <!-- 最上面 -->
            <Toolbar v-model:zoom="zoom" />

            <n-divider class="spliter-toolbar" />

            <!-- 左侧: 镜像棋盘 -->
            <Chessboard />

            <n-divider vertical class="spliter-middle" />

            <!-- 右侧: 分析面板 -->
            <Analyse />

            <n-layout-footer class="footer" position="absolute">
                <span>三炮十五兵走法提示工具 · 参考 atopx/chessboard 架构</span>
            </n-layout-footer>
        </n-message-provider>
    </n-dialog-provider>
</template>

<style scoped>
.footer {
    left: 13px;
    bottom: 5px;
    font-size: x-small;
}

.spliter-toolbar {
    position: absolute;
    width: 630px;
    top: 60px;
    left: 10px;
    z-index: 1;
}

.spliter-middle {
    position: absolute;
    left: 326px;
    top: 100px;
    height: 425px;
    z-index: 1;
}
</style>
