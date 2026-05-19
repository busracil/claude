"""
MAN Türkiye — Stok Optimizasyonu Karar Destek Arayüzü
======================================================
Çalıştırma:  streamlit run app/streamlit_app.py
"""

import os, sys, json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

st.set_page_config(page_title="MAN Stok Optimizasyonu", layout="wide",
                   initial_sidebar_state="expanded")

TL = lambda x: f"{x:,.0f} TL"
PCT = lambda x: f"%{x:.1f}"

# ─── Veri Yükleme (önbellek yok — her yenileme güncel veri okur) ──
def load_outputs():
    out = {}
    for key, path in [
        ("dashboard", C.DASHBOARD_PATH),
        ("opt",       C.OPTIMIZATION_PATH),
        ("champions", C.CHAMPIONS_PATH),
        ("forecasts", C.FORECASTS_PATH),
        ("method_metrics", os.path.join(C.OUTPUT_DIR, "method_metrics.parquet")),
        ("cost_maps",      os.path.join(C.OUTPUT_DIR, "cost_maps.parquet")),
    ]:
        if not os.path.exists(path):
            continue
        if path.endswith(".json"):
            out[key] = json.load(open(path, encoding="utf-8"))
        elif path.endswith(".csv"):
            out[key] = pd.read_csv(path)
        elif path.endswith(".parquet"):
            out[key] = pd.read_parquet(path)
    return out


@st.cache_data(show_spinner=False)
def load_history():
    df = pd.read_excel(C.EXCEL_PATH, sheet_name=C.SHEET_ML, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df[[C.COL_PART, C.COL_MONTH_IDX, C.COL_DATE, C.COL_TARGET_RAW, C.COL_SPLIT]]


def missing_data_error():
    st.error("Pipeline çıktıları bulunamadı. Önce `python pipeline.py` çalıştırın.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# SAYFA 1 — Genel Bakış
# ═══════════════════════════════════════════════════════════════════
def page_overview(data):
    st.title("Genel Bakış")
    d   = data.get("dashboard")
    opt = data.get("opt")
    if d is None or opt is None:
        missing_data_error()

    s = d["summary"]

    # KPI satırı
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Toplam Parça",       f"{s['total_parts']:,}")
    k2.metric("Optimize Edilen",    f"{s['optimized_parts']:,}")
    k3.metric("Ort. Tasarruf",      PCT(s['avg_pct_saving']))
    k4.metric("Ort. ML Servis",     PCT(s['avg_ml_service']*100))
    k5.metric("Ort. EOQ Servis",    PCT(s['avg_eoq_service']*100))

    st.divider()

    col_l, col_r = st.columns(2)

    # EOQ vs ML maliyet karşılaştırması
    with col_l:
        st.subheader("Toplam Maliyet: EOQ vs ML+SimPy")
        fig = go.Figure(go.Bar(
            x=["EOQ Klasik", "ML + SimPy"],
            y=[s["total_eoq_cost"], s["total_ml_cost"]],
            marker_color=["#c0392b", "#27ae60"],
            text=[TL(s["total_eoq_cost"]), TL(s["total_ml_cost"])],
            textposition="auto"))
        fig.update_layout(yaxis_title="Maliyet (TL)", height=340,
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Şampiyon yöntem dağılımı — pie
    with col_r:
        st.subheader("Şampiyon Yöntem Dağılımı")
        cd = d.get("champion_distribution", {})
        if cd:
            fig2 = px.pie(names=list(cd.keys()), values=list(cd.values()),
                          hole=0.4, height=340,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col_l2, col_r2 = st.columns(2)

    # Parça bazlı tasarruf histogramı
    with col_l2:
        st.subheader("Parça Bazlı Tasarruf Dağılımı (%)")
        if "pct_saving" in opt.columns:
            fig3 = px.histogram(opt, x="pct_saving", nbins=50, height=300,
                                labels={"pct_saving": "Tasarruf (%)"},
                                color_discrete_sequence=["#2980b9"])
            fig3.update_layout(yaxis_title="Parça Sayısı", margin=dict(t=10))
            st.plotly_chart(fig3, use_container_width=True)

    # ABC-XYZ Segment dağılımı
    with col_r2:
        st.subheader("ABC-XYZ Segment Dağılımı")
        sd = d.get("segment_distribution", {})
        if sd:
            seg_df = pd.DataFrame(list(sd.items()), columns=["Segment","Parça"])
            seg_df = seg_df.sort_values("Parça", ascending=False)
            fig4 = px.bar(seg_df, x="Segment", y="Parça", height=300,
                          color="Parça", color_continuous_scale="Blues")
            fig4.update_layout(margin=dict(t=10), coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Tüm Parça Sonuç Tablosu")
    cols = [c for c in [C.COL_PART, "Segment", "champion", "r_optimal",
            "Q_optimal", "SS_optimal", "ml_total_cost", "eoq_total_cost",
            "pct_saving", "ml_service_level"] if c in opt.columns]
    st.dataframe(opt[cols].rename(columns={
        "r_optimal": "r*", "Q_optimal": "Q*", "SS_optimal": "SS",
        "ml_total_cost": "ML Maliyet", "eoq_total_cost": "EOQ Maliyet",
        "pct_saving": "Tasarruf %", "ml_service_level": "Servis Düzeyi",
        "champion": "Şampiyon",
    }), use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════
# SAYFA 2 — Parça Detayı
# ═══════════════════════════════════════════════════════════════════
def page_part_detail(data, selected_part):
    st.title(f"Parça Detayı — {selected_part}")

    opt    = data.get("opt")
    fc     = data.get("forecasts")
    champs = data.get("champions")
    mm     = data.get("method_metrics")
    cmaps  = data.get("cost_maps")

    if opt is None or fc is None or champs is None:
        missing_data_error()

    row   = opt[opt[C.COL_PART] == selected_part]
    chrow = champs[champs[C.COL_PART] == selected_part]
    if row.empty or chrow.empty:
        st.warning("Bu parça için sonuç bulunamadı.")
        return
    row   = row.iloc[0]
    champ = chrow.iloc[0]["champion"]

    # ── Stok Parametreleri ──────────────────────────────────────
    st.subheader("Önerilen Stok Parametreleri (ML + SimPy)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Yeniden Sipariş Noktası r*", f"{row.get('r_optimal', 0):,.0f}")
    p2.metric("Sipariş Miktarı Q*",          f"{row.get('Q_optimal', 0):,.0f}")
    p3.metric("Emniyet Stoğu SS",            f"{row.get('SS_optimal', 0):,.0f}")
    p4.metric("Servis Düzeyi",               PCT(row.get('ml_service_level', 0)*100))

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("EOQ r",         f"{row.get('r_eoq', 0):,.0f}")
    p6.metric("EOQ Q",         f"{row.get('Q_eoq', 0):,.0f}")
    p7.metric("Şampiyon Model", champ)
    p8.metric("Maliyet Tasarrufu", PCT(row.get('pct_saving', 0)),
              TL(row.get('abs_saving', 0)))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Tahmin Grafiği", "📊 Hata Metrikleri", "💰 Maliyet Karşılaştırması", "🗺️ Grid Search Haritası"])

    # ── TAB 1: Tahmin Grafiği ──────────────────────────────────
    with tab1:
        hist = load_history()
        ph   = hist[hist[C.COL_PART] == selected_part].sort_values(C.COL_MONTH_IDX)
        pf   = fc[fc[C.COL_PART] == selected_part].sort_values(C.COL_MONTH_IDX)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ph[C.COL_MONTH_IDX], y=ph[C.COL_TARGET_RAW],
            mode="lines+markers", name="Gerçek Talep",
            line=dict(color="#2c3e50", width=2)))

        colors = px.colors.qualitative.Set1
        for i, (method, msub) in enumerate(pf.groupby("method")):
            is_champ = (method == champ)
            fig.add_trace(go.Scatter(
                x=msub[C.COL_MONTH_IDX], y=msub["pred"],
                mode="lines+markers",
                name=f"{method} {'★' if is_champ else ''}",
                line=dict(width=3 if is_champ else 1.5,
                          dash=None if is_champ else "dot",
                          color=colors[i % len(colors)])))

        fig.add_vrect(x0=30.5, x1=pf[C.COL_MONTH_IDX].max()+0.5,
                      fillcolor="lightyellow", opacity=0.3,
                      annotation_text="Test Ufku", annotation_position="top left")
        fig.update_layout(height=450, xaxis_title="Ay Sırası",
                          yaxis_title="Talep (adet)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ay 1-30: Eğitim verisi  |  Ay 31-36: Test ufku (sarı bölge)")

    # ── TAB 2: Hata Metrikleri ────────────────────────────────
    with tab2:
        if mm is not None:
            pm = mm[mm[C.COL_PART] == selected_part].copy()
            if not pm.empty:
                pm["Şampiyon"] = pm["method"].apply(lambda m: "★" if m == champ else "")
                pm = pm.rename(columns={"method": "Yöntem"})
                metric_cols = [c for c in ["MAE","RMSE","WAPE","sMAPE"] if c in pm.columns]

                # Metrik tablosu
                display_cols = ["Yöntem", "Şampiyon"] + metric_cols
                pm_disp = pm[display_cols].sort_values("WAPE").reset_index(drop=True)
                st.dataframe(pm_disp.style.highlight_min(
                    subset=metric_cols, color="#d4edda", axis=0),
                    use_container_width=True, height=300)

                # Çubuk grafik — WAPE
                col_l, col_r = st.columns(2)
                with col_l:
                    fig_w = px.bar(pm.sort_values("WAPE"), x="Yöntem", y="WAPE",
                                   title="WAPE (%)", height=320,
                                   color="WAPE", color_continuous_scale="RdYlGn_r")
                    fig_w.update_layout(margin=dict(t=30), coloraxis_showscale=False)
                    st.plotly_chart(fig_w, use_container_width=True)
                with col_r:
                    fig_s = px.bar(pm.sort_values("sMAPE"), x="Yöntem", y="sMAPE",
                                   title="sMAPE (%)", height=320,
                                   color="sMAPE", color_continuous_scale="RdYlGn_r")
                    fig_s.update_layout(margin=dict(t=30), coloraxis_showscale=False)
                    st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("Bu parça için metrik verisi bulunamadı.")
        else:
            st.info("method_metrics.parquet bulunamadı.")

    # ── TAB 3: Maliyet Karşılaştırması ────────────────────────
    with tab3:
        cost_items = ["Elde Tutma", "Sipariş", "Stoksuz Kalma", "Toplam"]
        ml_vals  = [row.get(k, 0) for k in
                    ["ml_holding_cost","ml_ordering_cost","ml_stockout_cost","ml_total_cost"]]
        eoq_vals = [row.get(k, 0) for k in
                    ["eoq_holding_cost","eoq_ordering_cost","eoq_stockout_cost","eoq_total_cost"]]

        fig_c = go.Figure()
        fig_c.add_bar(name="ML + SimPy", x=cost_items, y=ml_vals,
                      marker_color="#27ae60",
                      text=[TL(v) for v in ml_vals], textposition="auto")
        fig_c.add_bar(name="EOQ Klasik", x=cost_items, y=eoq_vals,
                      marker_color="#c0392b",
                      text=[TL(v) for v in eoq_vals], textposition="auto")
        fig_c.update_layout(barmode="group", height=400,
                            yaxis_title="Maliyet (TL)")
        st.plotly_chart(fig_c, use_container_width=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Mutlak Tasarruf (EOQ - ML)", TL(row.get("abs_saving", 0)))
        col_b.metric("Yüzde Tasarruf", PCT(row.get("pct_saving", 0)))

    # ── TAB 4: Grid Search Haritası ──────────────────────────
    with tab4:
        if cmaps is not None:
            cm = cmaps[cmaps[C.COL_PART] == selected_part]
            if not cm.empty:
                pivot = cm.pivot_table(index="r", columns="Q", values="cost")
                fig_h = px.imshow(pivot, aspect="auto", origin="lower",
                                  color_continuous_scale="RdYlGn_r",
                                  labels={"color": "Maliyet (TL)"},
                                  title="Grid Search — r × Q Maliyet Haritası")
                fig_h.update_layout(height=460)
                st.plotly_chart(fig_h, use_container_width=True)

                # Optimal nokta
                best = cm.loc[cm["cost"].idxmin()]
                st.success(f"Optimal: r={best['r']:,.0f}  Q={best['Q']:,.0f}  "
                           f"Maliyet={TL(best['cost'])}")
            else:
                st.info("Bu parça için Grid Search verisi yok.")
        else:
            st.info("cost_maps.parquet bulunamadı.")


# ═══════════════════════════════════════════════════════════════════
# SAYFA 3 — Model Karşılaştırması (Global)
# ═══════════════════════════════════════════════════════════════════
def page_models(data):
    st.title("Model Karşılaştırması — Tüm Parçalar")
    d  = data.get("dashboard")
    mm = data.get("method_metrics")
    if d is None:
        missing_data_error()

    # Genel metrikler tablosu
    st.subheader("Yöntem Bazlı Genel Hata Metrikleri")
    mo = d.get("method_overall_metrics", {})
    if mo:
        dfm = (pd.DataFrame(mo).T
               .reset_index()
               .rename(columns={"index": "Yöntem"})
               .sort_values("WAPE"))
        st.dataframe(dfm.style.highlight_min(
            subset=["MAE","RMSE","WAPE","sMAPE"], color="#d4edda", axis=0),
            use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            fig1 = px.bar(dfm.sort_values("WAPE"), x="Yöntem", y="WAPE",
                          title="WAPE (%) — düşük daha iyi", height=350,
                          color="WAPE", color_continuous_scale="RdYlGn_r")
            fig1.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig1, use_container_width=True)
        with col_r:
            fig2 = px.bar(dfm.sort_values("sMAPE"), x="Yöntem", y="sMAPE",
                          title="sMAPE (%) — düşük daha iyi", height=350,
                          color="sMAPE", color_continuous_scale="RdYlGn_r")
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Optuna doğrulama WAPE
    st.subheader("ML Modelleri — Optuna Doğrulama WAPE")
    tuning = d.get("tuning_info", {})
    if tuning:
        df_t = pd.DataFrame([{"Model": k, "Doğrulama WAPE (%)": round(v, 2)}
                              for k, v in tuning.items()])
        col_l2, col_r2 = st.columns(2)
        col_l2.dataframe(df_t, use_container_width=True)
        fig3 = px.bar(df_t.sort_values("Doğrulama WAPE (%)"),
                      x="Model", y="Doğrulama WAPE (%)", height=300,
                      color="Doğrulama WAPE (%)", color_continuous_scale="Blues_r")
        fig3.update_layout(coloraxis_showscale=False)
        col_r2.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Şampiyon + Segment dağılımı
    st.subheader("Dağılımlar")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        cd = d.get("champion_distribution", {})
        if cd:
            fig4 = px.pie(names=list(cd.keys()), values=list(cd.values()),
                          title="Şampiyon Yöntem", hole=0.4, height=320,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        sd = d.get("segment_distribution", {})
        if sd:
            fig5 = px.pie(names=list(sd.keys()), values=list(sd.values()),
                          title="ABC-XYZ Segment", hole=0.4, height=320,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig5, use_container_width=True)

    with col_c:
        mg = d.get("model_group_distribution", {})
        if mg:
            fig6 = px.pie(names=list(mg.keys()), values=list(mg.values()),
                          title="Model Strateji Grubu", hole=0.4, height=320,
                          color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig6, use_container_width=True)

    # Parça × Yöntem metrik tablosu (filtrelenebilir)
    if mm is not None:
        st.divider()
        st.subheader("Parça × Yöntem Metrik Tablosu")
        methods = sorted(mm["method"].unique().tolist()) if "method" in mm.columns else []
        sel_methods = st.multiselect("Yöntem filtrele", methods, default=methods)
        filtered = mm[mm["method"].isin(sel_methods)] if sel_methods else mm
        st.dataframe(filtered, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════
# ANA
# ═══════════════════════════════════════════════════════════════════
def main():
    data = load_outputs()

    # Sidebar
    st.sidebar.title("MAN Türkiye")
    st.sidebar.caption("ML + Simülasyon ile Dinamik Envanter Optimizasyonu")

    pages = ["🏠 Genel Bakış", "🔍 Parça Detayı", "📊 Model Karşılaştırması"]
    choice = st.sidebar.radio("Ekran", pages)

    # Parça seçici — tüm sayfalarda sidebar'da görünür
    opt = data.get("opt")
    selected_part = None
    if opt is not None and not opt.empty:
        part_list = sorted(opt[C.COL_PART].unique().tolist())
        st.sidebar.divider()
        st.sidebar.subheader("Parça Seç")

        search = st.sidebar.text_input("Ara (parça kodu)", "")
        filtered_parts = [p for p in part_list if search.upper() in str(p).upper()] if search else part_list
        if filtered_parts:
            selected_part = st.sidebar.selectbox(
                f"{len(filtered_parts)} parça", filtered_parts, label_visibility="collapsed")
        else:
            st.sidebar.info("Eşleşen parça yok.")

        # Sidebar'da seçili parçanın özet bilgisi
        if selected_part:
            row = opt[opt[C.COL_PART] == selected_part]
            if not row.empty:
                row = row.iloc[0]
                st.sidebar.caption(f"**Şampiyon:** {row.get('champion','—')}")
                st.sidebar.caption(f"**Segment:** {row.get('Segment','—')}")
                st.sidebar.caption(
                    f"r*={row.get('r_optimal',0):,.0f}  "
                    f"Q*={row.get('Q_optimal',0):,.0f}  "
                    f"SS={row.get('SS_optimal',0):,.0f}")
                st.sidebar.caption(f"Tasarruf: %{row.get('pct_saving',0):.1f}")

    if choice == "🏠 Genel Bakış":
        page_overview(data)
    elif choice == "🔍 Parça Detayı":
        if selected_part:
            page_part_detail(data, selected_part)
        else:
            st.info("Lütfen sol panelden bir parça seçin.")
    elif choice == "📊 Model Karşılaştırması":
        page_models(data)


if __name__ == "__main__":
    main()
