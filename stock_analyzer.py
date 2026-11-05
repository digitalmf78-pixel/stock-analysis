# stock_analyzer.py
# भारतीय शेअर बाजार विश्लेषण मॉडेल — मुख्य लॉजिक

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import dalal
import pandas_ta_classic as ta
import yfinance as yf
from transformers import pipeline

# price-action-lib ऐच्छिक आहे (अडचण आल्यास खालील फ्लॅग False करा)
try:
    from price_action_lib import PriceActionAnalyzer
    PRICE_ACTION_AVAILABLE = True
except Exception:
    PRICE_ACTION_AVAILABLE = False

# ------------------------------------------------------------
# 1. कॉन्फिगरेशन
# ------------------------------------------------------------
SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "ITC", "LT", "BHARTIARTL", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "TATAMOTORS"
]
HISTORY_DAYS = 90
SENTIMENT_MODEL = "Vansh180/FinBERT-India-v1"

# ------------------------------------------------------------
# 2. डेटा फेचिंग
# ------------------------------------------------------------
def fetch_ohlcv(symbol: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """dalal वापरून NSE ऐतिहासिक OHLCV डेटा."""
    from datetime import datetime, timedelta
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    raw = dalal.history(symbol, start, end)
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df

def fetch_live_quote(symbol: str) -> dict:
    try:
        return dalal.quote(symbol)
    except Exception:
        return {}

# ------------------------------------------------------------
# 3. टेक्निकल इंडिकेटर्स
# ------------------------------------------------------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=50, append=True)
    return df

# ------------------------------------------------------------
# 4. कँडलस्टिक व चार्ट पॅटर्न
# ------------------------------------------------------------
def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    if not PRICE_ACTION_AVAILABLE:
        # fallback: साधे पॅटर्न
        df["pattern_hammer"] = False
        df["setup_pin_bar"] = False
        df["near_support"] = False
        df["near_resistance"] = False
        return df
    try:
        analyzer = PriceActionAnalyzer()
        result = analyzer.fetch_all(df.copy())
        for col in ["pattern_hammer", "pattern_doji", "setup_pin_bar",
                    "near_support", "near_resistance", "structure_trend"]:
            if col in result.columns:
                df[col] = result[col]
    except Exception:
        df["pattern_hammer"] = False
        df["setup_pin_bar"] = False
        df["near_support"] = False
        df["near_resistance"] = False
    return df

# ------------------------------------------------------------
# 5. न्यूज सेंटिमेंट
# ------------------------------------------------------------
_sentiment_pipe = None

def get_sentiment_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            top_k=None
        )
    return _sentiment_pipe

def score_news_for_symbol(symbol: str, max_news: int = 5) -> float:
    try:
        pipe = get_sentiment_pipeline()
        ticker = yf.Ticker(f"{symbol}.NS")
        news = ticker.news[:max_news]
    except Exception:
        return 0.0
    if not news:
        return 0.0

    texts = [item.get("title", "") for item in news if item.get("title")]
    if not texts:
        return 0.0

    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    scores = []
    for text in texts:
        try:
            out = pipe(text[:512])[0]
            best = max(out, key=lambda x: x["score"])
            scores.append(label_map.get(best["label"], 0.0) * best["score"])
        except Exception:
            continue
    return float(np.mean(scores)) if scores else 0.0

# ------------------------------------------------------------
# 6. सिग्नल स्कोअरिंग
# ------------------------------------------------------------
def compute_technical_score(row: pd.Series) -> float:
    score = 0.0
    rsi = row.get("RSI_14")
    if pd.notna(rsi):
        if rsi < 30:
            score += 0.30 * 1.0
        elif rsi > 70:
            score += 0.30 * -1.0

    macd = row.get("MACD_12_26_9")
    signal = row.get("MACDs_12_26_9")
    if pd.notna(macd) and pd.notna(signal):
        score += 0.25 * (1.0 if macd > signal else -1.0)

    close = row.get("close")
    lower = row.get("BBL_20_2.0")
    upper = row.get("BBU_20_2.0")
    if pd.notna(close) and pd.notna(lower) and pd.notna(upper):
        if close <= lower:
            score += 0.20 * 1.0
        elif close >= upper:
            score += 0.20 * -1.0

    if row.get("pattern_hammer") or row.get("setup_pin_bar"):
        score += 0.15 * 1.0
    if row.get("near_support"):
        score += 0.15 * 0.5
    if row.get("near_resistance"):
        score += 0.15 * -0.5
    return float(np.clip(score, -1.0, 1.0))

def generate_signal(technical_score: float, sentiment_score: float) -> str:
    composite = 0.7 * technical_score + 0.3 * sentiment_score
    if composite >= 0.25:
        return "BUY"
    elif composite <= -0.25:
        return "SELL"
    return "HOLD"

# ------------------------------------------------------------
# 7. मुख्य विश्लेषण फंक्शन
# ------------------------------------------------------------
def analyze_symbol(symbol: str) -> dict:
    try:
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 30:
            return {"symbol": symbol, "error": "पुरेसा डेटा नाही"}

        df = add_technical_indicators(df)
        df = add_price_action_features(df)
        last = df.iloc[-1]

        tech_score = compute_technical_score(last)
        sent_score = score_news_for_symbol(symbol)
        signal = generate_signal(tech_score, sent_score)

        quote = fetch_live_quote(symbol)
        return {
            "symbol": symbol,
            "ltp": quote.get("ltp"),
            "technical_score": round(tech_score, 3),
            "sentiment_score": round(sent_score, 3),
            "signal": signal,
            "rsi": float(last.get("RSI_14")) if pd.notna(last.get("RSI_14")) else None,
            "macd": float(last.get("MACD_12_26_9")) if pd.notna(last.get("MACD_12_26_9")) else None,
            "patterns": {
                "hammer": bool(last.get("pattern_hammer", False)),
                "pin_bar": bool(last.get("setup_pin_bar", False)),
                "near_support": bool(last.get("near_support", False)),
                "near_resistance": bool(last.get("near_resistance", False)),
            },
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}