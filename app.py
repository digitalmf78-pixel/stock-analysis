# app.py — Streamlit वेबसाइट

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from stock_analyzer import (
    fetch_ohlcv,
    add_technical_indicators,
    add_price_action_features,
    analyze_symbol,
    SYMBOLS
)

st.set_page_config(
    page_title="भारतीय शेअर बाजार विश्लेषक",
    page_icon="📈",
    layout="wide"
)

st.title("📈 भारतीय शेअर बाजार विश्लेषक")
st.markdown("NSE/BSE स्टॉक्सचे टेक्निकल, पॅटर्न आणि न्यूज सेंटिमेंट विश्लेषण")

with st.sidebar:
    st.header("⚙️ सेटिंग्ज")
    selected = st.multiselect(
        "विश्लेषणासाठी स्टॉक्स निवडा",
        options=SYMBOLS,
        default=SYMBOLS[:3]
    )
    run = st.button("🚀 विश्लेषण सुरू करा", type="primary", use_container_width=True)

if run and selected:
    results = []
    progress = st.progress(0, text="विश्लेषण सुरू...")

    for i, sym in enumerate(selected):
        progress.progress((i + 1) / len(selected), text=f"{sym} चे विश्लेषण होत आहे...")
        results.append(analyze_symbol(sym))

    progress.empty()
    df_res = pd.DataFrame(results)

    st.subheader("📊 अंतिम सिग्नल सारांश")
    display_cols = ["symbol", "ltp", "technical_score", "sentiment_score", "signal"]
    available_cols = [c for c in display_cols if c in df_res.columns]

    if available_cols:
        st.dataframe(df_res[available_cols], use_container_width=True)
    else:
        st.warning("कोणतेही वैध परिणाम मिळाले नाहीत.")

    st.subheader("🔍 सविस्तर विश्लेषण")
    tabs = st.tabs([r.get("symbol", "?") for r in results])

    for tab, res in zip(tabs, results):
        with tab:
            if "error" in res:
                st.error(f"त्रुटी: {res['error']}")
                continue

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("LTP (₹)", res.get("ltp", "N/A"))
            col2.metric("टेक्निकल स्कोअर", res.get("technical_score", "N/A"))
            col3.metric("सेंटिमेंट स्कोअर", res.get("sentiment_score", "N/A"))
            col4.metric("सिग्नल", res.get("signal", "N/A"))

            try:
                df = fetch_ohlcv(res["symbol"])
                df = add_technical_indicators(df)
                df = add_price_action_features(df)
                last_60 = df.tail(60)

                fig = go.Figure(data=[
                    go.Candlestick(
                        x=last_60.index,
                        open=last_60["open"], high=last_60["high"],
                        low=last_60["low"], close=last_60["close"],
                        name="भाव"
                    )
                ])
                if "SMA_50" in last_60.columns:
                    fig.add_trace(go.Scatter(
                        x=last_60.index, y=last_60["SMA_50"],
                        mode="lines", name="SMA 50",
                        line=dict(color="orange")
                    ))

                fig.update_layout(
                    title=f"{res['symbol']} — मागील 60 दिवसांचा चार्ट",
                    xaxis_title="तारीख", yaxis_title="भाव (₹)",
                    xaxis_rangeslider_visible=False, height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"चार्ट तयार करता आला नाही: {e}")

            patterns = res.get("patterns", {})
            if patterns:
                st.write("**डिटेक्टेड पॅटर्न:**")
                cols = st.columns(len(patterns))
                for col, (name, val) in zip(cols, patterns.items()):
                    icon = "✅" if val else "❌"
                    col.write(f"{icon} {name.replace('_', ' ').title()}")

else:
    if not selected:
        st.info("साइडबारमधून किमान एक स्टॉक निवडा.")
    else:
        st.info("सुरू करण्यासाठी 'विश्लेषण सुरू करा' बटण दाबा.")