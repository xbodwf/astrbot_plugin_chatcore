/* ChatCore Dashboard Pages frontend.
 *
 * No build step: a plain ESM module that imports the self-contained MUI v9
 * bundle from ./vendor/mui.full.js and talks to the backend through the
 * AstrBotPluginPage bridge (window.AstrBotPluginPage).
 *
 * Two tabs:
 *   监控 - live monitoring (attention / context / emotion / counts).
 *   管理 - management (profiles / memories / emoji / expression styles).
 */

import * as MUI from "./vendor/mui.full.js";

const { React, ReactDOMClient } = MUI;
const {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  CssVarsProvider,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Snackbar,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  TextField,
  Typography,
  createTheme,
} = MUI;

const h = React.createElement;

const theme = createTheme({
  colorSchemes: { light: true, dark: true },
});

const bridge = window.AstrBotPluginPage;

/* ---------- helpers ---------- */

function fmtTime(ts) {
  if (!ts) return "-";
  const date = new Date(ts * 1000);
  return isNaN(date.getTime()) ? String(ts) : date.toLocaleString();
}

function fmtCount(n) {
  return typeof n === "number" ? String(n) : String(n ?? 0);
}

/* ---------- mode sync: data-theme -> MUI color scheme ---------- */

function ModeSync() {
  const { setMode } = MUI.useColorScheme();
  React.useEffect(() => {
    const apply = () => {
      const themeName = document.documentElement.getAttribute("data-theme");
      setMode(themeName === "dark" ? "dark" : "light");
    };
    apply();
    const off = bridge.onContext(apply);
    const observer = new MutationObserver(apply);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => {
      off();
      observer.disconnect();
    };
  }, [setMode]);
  return null;
}

/* ---------- generic load-state hook ---------- */

function useLoad(fetcher, deps) {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(true);
  const reload = React.useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetcher();
      setData(res);
    } catch (err) {
      setError(err && err.message ? err.message : String(err));
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps || []);
  React.useEffect(() => {
    reload();
  }, [reload]);
  return { data, error, loading, reload };
}

function ErrorBanner({ error }) {
  if (!error) return null;
  return h(
    Alert,
    { severity: "error", sx: { mb: 2 } },
    error,
  );
}

function LoadingBox() {
  return h(
    Box,
    { sx: { display: "flex", justifyContent: "center", py: 6 } },
    h(CircularProgress),
  );
}

/* ---------- stat cards ---------- */

function StatCard({ label, value }) {
  return h(
    Card,
    { variant: "outlined", sx: { height: "100%" } },
    h(
      CardContent,
      null,
      h(Typography, { variant: "body2", color: "text.secondary" }, label),
      h(Typography, { variant: "h5", sx: { fontWeight: 600, mt: 0.5 } }, fmtCount(value)),
    ),
  );
}

/* ---------- monitoring tables ---------- */

function AttentionTable({ rows }) {
  const list = rows || [];
  return h(
    TableContainer,
    { component: Paper, variant: "outlined" },
    h(
      Table,
      { size: "small" },
      h(
        TableHead,
        null,
        h(
          TableRow,
          null,
          h(TableCell, null, "会话"),
          h(TableCell, null, "冒泡概率"),
          h(TableCell, null, "冷却中"),
          h(TableCell, null, "软触发落空"),
          h(TableCell, null, "他人消息数"),
          h(TableCell, null, "最后互动"),
        ),
      ),
      h(
        TableBody,
        null,
        list.length === 0
          ? h(
              TableRow,
              null,
              h(
                TableCell,
                { colSpan: 6, align: "center" },
                "暂无活跃会话",
              ),
            )
          : list.map((row) =>
              h(
                TableRow,
                { key: row.group_id },
                h(TableCell, null, row.group_id),
                h(TableCell, null, `${(row.probability * 100).toFixed(1)}%`),
                h(
                  TableCell,
                  null,
                  row.in_cooldown
                    ? h(Chip, { label: "冷却", color: "error", size: "small" })
                    : h(Chip, { label: "正常", color: "success", size: "small" }),
                ),
                h(TableCell, null, fmtCount(row.soft_misses)),
                h(TableCell, null, fmtCount(row.others_recent)),
                h(TableCell, null, fmtTime(row.last_interaction_ts)),
              ),
            ),
      ),
    ),
  );
}

function ContextTable({ rows }) {
  const list = rows || [];
  return h(
    TableContainer,
    { component: Paper, variant: "outlined" },
    h(
      Table,
      { size: "small" },
      h(
        TableHead,
        null,
        h(
          TableRow,
          null,
          h(TableCell, null, "会话"),
          h(TableCell, null, "消息数"),
          h(TableCell, null, "压缩历史"),
          h(TableCell, null, "摘要"),
          h(TableCell, null, "摘要长度"),
        ),
      ),
      h(
        TableBody,
        null,
        list.length === 0
          ? h(
              TableRow,
              null,
              h(
                TableCell,
                { colSpan: 5, align: "center" },
                "暂无会话上下文",
              ),
            )
          : list.map((row) =>
              h(
                TableRow,
                { key: row.conv_id },
                h(TableCell, null, row.conv_id),
                h(TableCell, null, fmtCount(row.messages)),
                h(TableCell, null, fmtCount(row.older_count)),
                h(
                  TableCell,
                  null,
                  row.has_summary
                    ? h(Chip, { label: "有", color: "success", size: "small" })
                    : h(Chip, { label: "无", size: "small" }),
                ),
                h(TableCell, null, fmtCount(row.summary_len)),
              ),
            ),
      ),
    ),
  );
}

function EmotionList({ rows }) {
  const list = rows || [];
  if (list.length === 0) {
    return h(Typography, { variant: "body2", color: "text.secondary" }, "暂无情绪状态");
  }
  return h(
    Stack,
    { direction: "column", spacing: 1 },
    list.map((row) =>
      h(
        Paper,
        {
          key: row.conv_id,
          variant: "outlined",
          sx: { p: 1.5, display: "flex", alignItems: "center", gap: 1.5 },
        },
        h(Chip, { label: row.mood || "neutral", color: "primary", size: "small" }),
        h(Typography, { variant: "body2", sx: { flex: 1, minWidth: 0 } }, row.conv_id),
        h(Typography, { variant: "caption", color: "text.secondary" }, `特质: ${row.trait || "-"}`),
      ),
    ),
  );
}

/* ---------- management panels ---------- */

function ProfilesPanel() {
  const { data, error, loading, reload } = useLoad(() => bridge.apiGet("profiles"));
  const [busy, setBusy] = React.useState(false);
  const profiles = data?.profiles || [];

  const doDelete = async (personId) => {
    setBusy(true);
    try {
      await bridge.apiPost("profiles/delete", { person_id: personId });
      await reload();
    } finally {
      setBusy(false);
    }
  };

  if (loading) return h(LoadingBox);
  return h(
    Box,
    null,
    h(ErrorBanner, { error }),
    profiles.length === 0
      ? h(Typography, { variant: "body2", color: "text.secondary" }, "暂无人物画像")
      : h(
          Stack,
          { direction: "column", spacing: 2 },
          profiles.map((profile) =>
            h(
              Card,
              { key: profile.person_id, variant: "outlined" },
              h(
                CardContent,
                null,
                h(
                  Stack,
                  { direction: "row", alignItems: "center", spacing: 1 },
                  h(
                    Typography,
                    { variant: "subtitle1", sx: { fontWeight: 600 } },
                    profile.nickname || profile.person_id,
                  ),
                  h(
                    Button,
                    {
                      size: "small",
                      color: "error",
                      onClick: () => doDelete(profile.person_id),
                      disabled: busy,
                      sx: { ml: "auto" },
                    },
                    "删除",
                  ),
                ),
                h(Typography, { variant: "caption", color: "text.secondary" }, profile.person_id),
                h(Divider, { sx: { my: 1 } }),
                (profile.facts || []).map((fact, idx) =>
                  h(Typography, { key: idx, variant: "body2", sx: { my: 0.25 } }, `· ${fact}`),
                ),
              ),
            ),
          ),
        ),
  );
}

function MemoriesPanel() {
  const { data, error, loading, reload } = useLoad(() => bridge.apiGet("memories"));
  const [busy, setBusy] = React.useState(false);
  const [groupFilter, setGroupFilter] = React.useState("");
  const [visibleCount, setVisibleCount] = React.useState(20);
  const memories = data?.memories || [];

  const doDelete = async (index) => {
    setBusy(true);
    try {
      await bridge.apiPost("memories/delete", { index });
      await reload();
    } finally {
      setBusy(false);
    }
  };

  const kw = groupFilter.trim().toLowerCase();
  const filtered = kw
    ? memories.filter((entry) =>
        (entry.tags || []).some((tag) => String(tag).toLowerCase().includes(kw)),
      )
    : memories;
  const visible = filtered.slice(0, visibleCount);
  const showMore = filtered.length > visible.length;

  if (loading) return h(LoadingBox);
  return h(
    Box,
    null,
    h(ErrorBanner, { error }),
    h(
      Stack,
      { direction: "row", spacing: 1, alignItems: "center", mb: 1.5 },
      h(TextField, {
        size: "small",
        variant: "outlined",
        placeholder: "按群/会话过滤（如 GroupMessage:384128966）",
        value: groupFilter,
        onChange: (e) => {
          setGroupFilter(e.target.value);
          setVisibleCount(20);
        },
        sx: { flex: 1, maxWidth: 420 },
      }),
      h(Typography, { variant: "caption", color: "text.secondary" }, `共 ${filtered.length} 条`),
    ),
    filtered.length === 0
      ? h(Typography, { variant: "body2", color: "text.secondary" }, "暂无记忆片段")
      : h(
          Stack,
          { direction: "column", spacing: 1 },
          visible.map((entry) =>
            h(
              Paper,
              { key: entry.index, variant: "outlined", sx: { p: 1.5 } },
              h(
                Stack,
                { direction: "row", alignItems: "flex-start", spacing: 1 },
                h(
                  Box,
                  { sx: { flex: 1, minWidth: 0 } },
                  h(Typography, { variant: "body2" }, entry.text),
                  h(
                    Stack,
                    { direction: "row", spacing: 0.5, mt: 0.75 },
                    (entry.tags || []).map((tag) =>
                      h(Chip, { key: tag, label: tag, size: "small", variant: "outlined" }),
                    ),
                  ),
                  h(Typography, { variant: "caption", color: "text.secondary" }, fmtTime(entry.ts)),
                ),
                h(
                  Button,
                  {
                    size: "small",
                    color: "error",
                    onClick: () => doDelete(entry.index),
                    disabled: busy,
                  },
                  "删除",
                ),
              ),
            ),
          ),
        ),
    showMore &&
      h(
        Button,
        { size: "small", variant: "text", onClick: () => setVisibleCount((n) => n + 20) },
        "加载更多",
      ),
  );
}

function EmojiThumb({ emojiId }) {
  const [src, setSrc] = React.useState("");
  React.useEffect(() => {
    let cancelled = false;
    bridge
      .apiGet(`emojis/${encodeURIComponent(emojiId)}/image/data`)
      // The parent bridge unwraps `response.data.data`, so this resolves to
      // the data-URI string directly.
      .then((uri) => {
        if (!cancelled && uri && typeof uri === "string") setSrc(uri);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [emojiId]);
  if (!src) return h(Box, { sx: { width: 64, height: 64, borderRadius: 1, bgcolor: "action.hover" } });
  return h("img", { src, alt: emojiId, style: { width: 64, height: 64, objectFit: "cover", borderRadius: 4 } });
}

function EmojisPanel() {
  const { data, error, loading, reload } = useLoad(() => bridge.apiGet("emojis"));
  const [busy, setBusy] = React.useState(false);
  const [editing, setEditing] = React.useState(null);
  const [importing, setImporting] = React.useState(false);
  const fileInputRef = React.useRef(null);
  const emojis = data?.emojis || [];

  const doDelete = async (emojiId) => {
    setBusy(true);
    try {
      await bridge.apiPost("emojis/delete", { emoji_id: emojiId });
      await reload();
    } finally {
      setBusy(false);
    }
  };

  const doImport = async (file) => {
    if (!file) return;
    setImporting(true);
    try {
      await bridge.upload("emojis/import", file);
      await reload();
    } catch (e) {
      console.error("emoji import failed", e);
    } finally {
      setImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const openEdit = (emoji) => setEditing({ ...emoji, tagsText: (emoji.tags || []).join(", ") });
  const saveEdit = async () => {
    if (!editing) return;
    setBusy(true);
    try {
      await bridge.apiPost("emojis/update", {
        emoji_id: editing.emoji_id,
        category: editing.category || "",
        tags: editing.tagsText
          .split(/[,，]/)
          .map((t) => t.trim())
          .filter(Boolean),
      });
      setEditing(null);
      await reload();
    } finally {
      setBusy(false);
    }
  };

  if (loading) return h(LoadingBox);
  return h(
    Box,
    null,
    h(ErrorBanner, { error }),
    h(
      Stack,
      { direction: "row", spacing: 1, alignItems: "center", mb: 1.5 },
      h(
        Button,
        {
          size: "small",
          variant: "tonal",
          disabled: importing,
          onClick: () => fileInputRef.current?.click(),
        },
        importing ? "导入中…" : "导入表情包",
      ),
      h(Typography, { variant: "caption", color: "text.secondary" }, `共 ${emojis.length} 个`),
      h("input", {
        ref: fileInputRef,
        type: "file",
        accept: "image/*",
        style: { display: "none" },
        onChange: (e) => doImport(e.target.files?.[0]),
      }),
    ),
    emojis.length === 0
      ? h(Typography, { variant: "body2", color: "text.secondary" }, "暂无表情包")
      : h(
          Stack,
          { direction: "column", spacing: 1.5 },
          emojis.map((emoji) =>
            h(
              Paper,
              { key: emoji.emoji_id, variant: "outlined", sx: { p: 1.5 } },
              h(
                Stack,
                { direction: "row", spacing: 1.5, alignItems: "flex-start" },
                h(EmojiThumb, { emojiId: emoji.emoji_id }),
                h(
                  Box,
                  { sx: { flex: 1, minWidth: 0 } },
                  h(
                    Stack,
                    { direction: "row", spacing: 1, alignItems: "center" },
                    h(
                      Chip,
                      { label: emoji.category || "未分类", color: "primary", size: "small" },
                    ),
                    h(
                      Typography,
                      { variant: "caption", color: "text.secondary" },
                      `编号 ${emoji.emoji_id}`,
                    ),
                  ),
                  (emoji.tags || []).length > 0 &&
                    h(
                      Stack,
                      { direction: "row", spacing: 0.5, mt: 0.5, flexWrap: "wrap" },
                      emoji.tags.map((tag) =>
                        h(Chip, { key: tag, label: tag, size: "small", variant: "outlined" }),
                      ),
                    ),
                  emoji.source_text &&
                    h(
                      Typography,
                      { variant: "body2", sx: { mt: 0.5, color: "text.secondary" } },
                      `原消息: ${emoji.source_text}`,
                    ),
                  h(
                    Typography,
                    { variant: "caption", color: "text.secondary" },
                    `来源: ${emoji.source_group || "未知群"} / ${emoji.source_sender || "?"} · 使用 ${fmtCount(emoji.usage_count)} 次 · ${fmtTime(emoji.collected_at)}`,
                  ),
                ),
                h(
                  Stack,
                  { direction: "column", spacing: 0.5 },
                  h(
                    Button,
                    { size: "small", onClick: () => openEdit(emoji), disabled: busy },
                    "编辑",
                  ),
                  h(
                    Button,
                    { size: "small", color: "error", onClick: () => doDelete(emoji.emoji_id), disabled: busy },
                    "删除",
                  ),
                ),
              ),
            ),
          ),
        ),
    h(
      Dialog,
      { open: !!editing, onClose: () => setEditing(null), maxWidth: "sm", fullWidth: true },
      editing &&
        h(React.Fragment, null,
          h(DialogTitle, null, "编辑表情包"),
          h(
            DialogContent,
            null,
            h(
              FormControl,
              { fullWidth: true, size: "small", sx: { mt: 1 } },
              h(InputLabel, null, "分类"),
              h(
                Select,
                {
                  value: editing.category || "",
                  label: "分类",
                  onChange: (e) => setEditing({ ...editing, category: e.target.value }),
                },
                ["开心", "嘲讽", "敷衍", "震惊", "生气", "可爱", "疑问", "委屈", "无语", "其他"].map(
                  (c) => h(MenuItem, { key: c, value: c }, c),
                ),
              ),
            ),
            h(
              TextField,
              {
                fullWidth: true,
                size: "small",
                label: "标签（逗号分隔）",
                value: editing.tagsText || "",
                onChange: (e) => setEditing({ ...editing, tagsText: e.target.value }),
                sx: { mt: 2 },
              },
            ),
            h(
              Typography,
              { variant: "caption", color: "text.secondary", sx: { mt: 1, display: "block" } },
              `来源语境: ${editing.source_context || "无"}`,
            ),
          ),
          h(
            DialogActions,
            null,
            h(Button, { onClick: () => setEditing(null) }, "取消"),
            h(Button, { variant: "contained", onClick: saveEdit, disabled: busy }, "保存"),
          ),
        ),
    ),
  );
}

function ExpressionsPanel() {
  const { data, error, loading, reload } = useLoad(() => bridge.apiGet("expressions"));
  const [busy, setBusy] = React.useState(false);
  const expressions = data?.expressions || [];

  const doDelete = async (groupId) => {
    setBusy(true);
    try {
      await bridge.apiPost("expressions/delete", { group_id: groupId });
      await reload();
    } finally {
      setBusy(false);
    }
  };

  if (loading) return h(LoadingBox);
  return h(
    Box,
    null,
    h(ErrorBanner, { error }),
    expressions.length === 0
      ? h(Typography, { variant: "body2", color: "text.secondary" }, "暂无表达风格")
      : h(
          Stack,
          { direction: "column", spacing: 1 },
          expressions.map((style) =>
            h(
              Paper,
              { key: style.group_id, variant: "outlined", sx: { p: 1.5 } },
              h(
                Stack,
                { direction: "row", alignItems: "center", spacing: 1 },
                h(Typography, { variant: "subtitle2" }, style.group_id),
                h(
                  Button,
                  {
                    size: "small",
                    color: "error",
                    onClick: () => doDelete(style.group_id),
                    disabled: busy,
                    sx: { ml: "auto" },
                  },
                  "删除",
                ),
              ),
              (style.jargon || []).length > 0 &&
                h(
                  Stack,
                  { direction: "row", spacing: 0.5, mt: 0.75, flexWrap: "wrap" },
                  style.jargon.map((item) =>
                    h(Chip, {
                      key: item.term,
                      label: item.term,
                      size: "small",
                      variant: "outlined",
                      title: `${item.meaning || ""}${item.example ? `（例: ${item.example}）` : ""}`,
                    }),
                  ),
                ),
              (style.samples || []).slice(0, 3).map((sample, idx) =>
                h(Typography, { key: idx, variant: "body2", color: "text.secondary", sx: { mt: 0.5 } }, sample),
              ),
            ),
          ),
        ),
  );
}

/* ---------- tabs ---------- */

function MonitorTab() {
  const { data, error, loading, reload } = useLoad(() => bridge.apiGet("stats"), []);
  const [auto, setAuto] = React.useState(true);
  React.useEffect(() => {
    if (!auto) return () => {};
    const timer = setInterval(() => reload(), 8000);
    return () => clearInterval(timer);
  }, [auto, reload]);

  if (loading && !data) return h(LoadingBox);
  return h(
    Box,
    null,
    h(ErrorBanner, { error }),
    h(
      Stack,
      { direction: "row", spacing: 2, mb: 2 },
      h(StatCard, { label: "人物画像", value: data?.profile_count }),
      h(StatCard, { label: "记忆片段", value: data?.memory_count }),
      h(StatCard, { label: "表情包", value: data?.emoji_count }),
      h(StatCard, { label: "表达风格", value: data?.expression_count }),
      h(StatCard, { label: "撤回取消", value: data?.recalls_cancelled }),
    ),
    h(
      Button,
      { onClick: reload, variant: "outlined", size: "small", sx: { mb: 1.5 } },
      "刷新",
    ),
    h(Typography, { variant: "h6", sx: { mt: 1, mb: 1 } }, "注意力 / 退避"),
    h(AttentionTable, { rows: data?.attention }),
    h(Typography, { variant: "h6", sx: { mt: 3, mb: 1 } }, "上下文窗口"),
    h(ContextTable, { rows: data?.context }),
    h(Typography, { variant: "h6", sx: { mt: 3, mb: 1 } }, "情绪状态"),
    h(EmotionList, { rows: data?.emotion }),
  );
}

function ManageTab() {
  const [tab, setTab] = React.useState("profiles");
  const panel = {
    profiles: h(ProfilesPanel),
    memories: h(MemoriesPanel),
    emojis: h(EmojisPanel),
    expressions: h(ExpressionsPanel),
  }[tab];
  return h(
    Box,
    null,
    h(
      Tabs,
      { value: tab, onChange: (_, v) => setTab(v), variant: "scrollable" },
      h(Tab, { label: "画像", value: "profiles" }),
      h(Tab, { label: "记忆", value: "memories" }),
      h(Tab, { label: "表情包", value: "emojis" }),
      h(Tab, { label: "表达风格", value: "expressions" }),
    ),
    h(Box, { sx: { mt: 2 } }, panel),
  );
}

function App() {
  const [tab, setTab] = React.useState("monitor");
  return h(
    Box,
    { sx: { minHeight: "100vh" } },
    h(ModeSync),
    h(
      Box,
      {
        sx: {
          px: 2,
          pt: 2,
          pb: 1,
          borderBottom: 1,
          borderColor: "divider",
          position: "sticky",
          top: 0,
          bgcolor: "background.default",
          zIndex: 10,
        },
      },
      h(Typography, { variant: "h6", sx: { fontWeight: 600 } }, "ChatCore 管理面板"),
      h(
        Tabs,
        { value: tab, onChange: (_, v) => setTab(v) },
        h(Tab, { label: "监控", value: "monitor" }),
        h(Tab, { label: "管理", value: "manage" }),
      ),
    ),
    h(Box, { sx: { p: 2, maxWidth: 1000, margin: "0 auto" } }, tab === "monitor" ? h(MonitorTab) : h(ManageTab)),
  );
}

/* ---------- boot ---------- */

async function boot() {
  try {
    await bridge.ready();
  } catch (err) {
    console.warn("ChatCore page bridge not ready:", err);
  }
  ReactDOMClient.createRoot(document.getElementById("root")).render(
    h(CssVarsProvider, { theme }, h(App)),
  );
}

boot();
