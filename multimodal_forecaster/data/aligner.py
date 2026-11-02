"""
Phase 2: Static Data Aggregation & 4-Modality Alignment (FR1).

Ingests raw historical datasets from the `data/` folder across 4 core channels:
- Index 0: Price Time-Series (`price.csv` / `{asset}_price.csv` / `{asset}_ohlcv.csv`)
           Raw OHLCV -> computes 8 channels [Open, High, Low, Close, Volume, EMA, RSI, BB_Width]
           over lookback window h=30 -> X_OHLCV in R^{h x 8}.
- Index 1: Social Sentiment (`sentiment.csv` / `{asset}_sentiment.csv`)
           Accepts either raw text comments (encoded via FinBERT into [Prob_pos, Prob_neg, Prob_neut])
           or pre-computed polarity columns over lookback window h=30 -> X_sentiment in R^{h x 3}.
- Index 2: Expert Financial News (`news.csv` / `{asset}_news.csv` / `{asset}_news.npy`)
           Accepts either raw news headlines/articles (encoded via DistilBERT into R^{256})
           or pre-extracted 256-d vectors over sliding delay window d=5 -> X_Expert in R^{d x 256}.
- Index 3: Macro Inflation Indicators (`macro.csv` / `inflation.csv` / `{asset}_macro.csv`)
           Accepts monthly or sparse daily CPI/inflation records (appearing once a month /
           every 30 steps with intermediate NaNs), computes temporal latency delta_T, and
           forward-fills for encoder ingestion -> X_macro in R^{k}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertModel,
)


PRICE_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "EMA",
    "RSI",
    "BB_Width",
]

SENTIMENT_COLUMNS = ["Prob_pos", "Prob_neg", "Prob_neut"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class AlignedDatasetBundle:
    """Container holding temporally aligned 4-modality arrays and metadata."""

    X_OHLCV: np.ndarray          # [N, h, 8]
    X_sentiment: np.ndarray      # [N, h, 3] FinBERT polarities
    X_Expert: np.ndarray         # [N, d, 256] DistilBERT embeddings
    X_macro: np.ndarray          # [N, k] (forward-filled for numerical stability)
    raw_macro_series: np.ndarray # [T] raw daily macro series containing NaNs (1 valid every 30 rows)
    delta_T: np.ndarray          # [N, 4] temporal ingestion latency vector
    S_state: np.ndarray          # [N, V] market state volatility vector (V=3)
    a_close: np.ndarray          # [N, 1] rolling lag-based closing average baseline
    y_target: np.ndarray         # [N, 1] next-day absolute closing price P^{(t+1)}
    P_current: np.ndarray        # [N, 1] current-day absolute closing price P^{(t)}
    timestamps: pd.DatetimeIndex # [N] aligned evaluation dates
    asset_name: str = "Gold"


class MultimodalDataAligner:
    """
    Implements FR1: Multi-Stream Ingestion and Temporal Alignment from raw files
    stored in the `data/` folder across Price (h=30, 8 features), FinBERT Sentiment
    (h=30, 3 features), DistilBERT Expert News (d=5, 256 features), and Monthly
    Macro Inflation (every 30 steps, intermediate rows NaN).
    """

    FINBERT_MODEL_NAME = "ProsusAI/finbert"
    DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

    def __init__(
        self,
        lookback_h: int = 30,
        news_delay_d: int = 5,
        news_embed_dim: int = 256,
        macro_dim: int = 1,
        v_indices: int = 3,
        macro_period_steps: int = 30,
        start_date: str = "2019-01-01",
        end_date: str = "2023-12-31",
        load_pretrained_nlp: bool = False,
    ) -> None:
        self.lookback_h = lookback_h
        self.news_delay_d = news_delay_d
        self.news_embed_dim = news_embed_dim
        self.macro_dim = macro_dim
        self.v_indices = v_indices
        self.macro_period_steps = macro_period_steps
        self.start_date = start_date
        self.end_date = end_date
        self.load_pretrained_nlp = load_pretrained_nlp

    # =========================================================================
    # NLP Feature Extraction Helpers (FinBERT & DistilBERT)
    # =========================================================================
    @classmethod
    def extract_finbert_sentiment(
        cls,
        texts: List[str],
        load_pretrained: bool = False,
        batch_size: int = 64,
        device: Optional[torch.device | str] = None,
    ) -> np.ndarray:
        """
        Extracts FinBERT polarity arrays [Prob_pos, Prob_neg, Prob_neut] in R^{N x 3}
        from raw social sentiment text strings using Hugging Face `ProsusAI/finbert`
        (or offline FinBERT `BertForSequenceClassification` when offline).
        """
        if len(texts) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_pretrained:
            tokenizer = AutoTokenizer.from_pretrained(cls.FINBERT_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                cls.FINBERT_MODEL_NAME, num_labels=3
            ).to(device)
        else:
            torch.manual_seed(101)
            config = BertConfig(
                vocab_size=30522,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=256,
                num_labels=3,
            )
            model = BertForSequenceClassification(config).to(device)
            tokenizer = None

        model.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [str(t) if pd.notna(t) else "" for t in texts[start : start + batch_size]]
                if tokenizer is not None:
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(device)
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded["attention_mask"]
                else:
                    max_len = 24
                    ids = np.full((len(batch_texts), max_len), 101, dtype=np.int64)
                    for i, txt in enumerate(batch_texts):
                        words = txt.split()[:max_len]
                        for j, w in enumerate(words):
                            ids[i, j] = (sum(ord(c) * (idx + 1) for idx, c in enumerate(w)) % 29000) + 105
                    input_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    @classmethod
    def extract_distilbert_news_embeddings(
        cls,
        headlines: List[str],
        d_embed: int = 256,
        max_length: int = 256,
        load_pretrained: bool = False,
        batch_size: int = 64,
        device: Optional[torch.device | str] = None,
    ) -> np.ndarray:
        """
        Extracts contextual DistilBERT feature representations in R^{N x 256}
        from raw financial news headlines/articles using Hugging Face `distilbert-base-uncased`
        with tokenization sequence parameter `max_length=256` (not 512).
        """
        if len(headlines) == 0:
            return np.zeros((0, d_embed), dtype=np.float32)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_pretrained:
            tokenizer = AutoTokenizer.from_pretrained(cls.DISTILBERT_MODEL_NAME)
            model = AutoModel.from_pretrained(cls.DISTILBERT_MODEL_NAME).to(device)
            out_dim = model.config.dim
        else:
            torch.manual_seed(202)
            config = DistilBertConfig(
                vocab_size=30522,
                dim=d_embed,
                n_layers=2,
                n_heads=4,
                hidden_dim=d_embed,
                max_position_embeddings=max_length,
            )
            model = DistilBertModel(config).to(device)
            out_dim = d_embed
            tokenizer = None

        proj = (
            torch.nn.Identity().to(device)
            if out_dim == d_embed
            else torch.nn.Linear(out_dim, d_embed).to(device)
        )
        model.eval()
        all_vecs = []

        with torch.no_grad():
            for start in range(0, len(headlines), batch_size):
                batch_texts = [str(t) if pd.notna(t) else "" for t in headlines[start : start + batch_size]]
                if tokenizer is not None:
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(device)
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded["attention_mask"]
                else:
                    seq_cap = min(max_length, 256)
                    max_words = max(1, min(seq_cap, max(len(t.split()) for t in batch_texts) if batch_texts else 1))
                    ids = np.full((len(batch_texts), max_words), 101, dtype=np.int64)
                    for i, txt in enumerate(batch_texts):
                        words = txt.split()[:seq_cap]
                        for j, w in enumerate(words):
                            ids[i, j] = (sum(ord(c) * (idx + 1) for idx, c in enumerate(w)) % 29000) + 105
                    input_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)

                hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                cls_vec = proj(hidden[:, 0, :])
                all_vecs.append(cls_vec.cpu().numpy().astype(np.float32))

        return np.concatenate(all_vecs, axis=0)

    # =========================================================================
    # Raw Dataset Readers for the 4 Modalities in `data/`
    # =========================================================================
    @staticmethod
    def _parse_date_index(df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes a DataFrame so its index is a sorted daily DatetimeIndex."""
        df = df.copy()
        date_candidates = [
            "date",
            "created_utc",
            "datetime",
            "timestamp",
            "time",
            "day",
            "observation_date",
        ]
        col_map = {str(c).strip().lower(): c for c in df.columns}
        matched_col = None
        for cand in date_candidates:
            if cand in col_map:
                matched_col = col_map[cand]
                break

        if matched_col is not None:
            raw_s = df[matched_col].astype(str).str.strip()
            valid_samples = raw_s[raw_s != "nan"]
            dayfirst = True
            if len(valid_samples) > 0:
                first_val = valid_samples.iloc[0]
                if len(first_val) >= 10 and first_val[:4].isdigit() and first_val[4] == "-":
                    dayfirst = False
            df[matched_col] = pd.to_datetime(
                df[matched_col], format="mixed", dayfirst=dayfirst, errors="coerce"
            )
            df = df[df[matched_col].notna()].set_index(matched_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, format="mixed", errors="coerce")
            df = df[df.index.notna()]

        df.index = df.index.tz_localize(None).normalize()
        df = df.sort_index()
        return df

    @staticmethod
    def compute_technical_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the 8 structural price channels from raw OHLCV data:
        1. Open, 2. High, 3. Low, 4. Close, 5. Volume,
        6. EMA (14-day), 7. RSI (14-day), 8. Bollinger Band Width (20-day).
        """
        df = df_ohlcv.copy()
        # Case-insensitive column normalization for Open, High, Low, Close, Volume
        rename_map = {}
        for col in df.columns:
            low = str(col).strip().lower()
            if low == "open":
                rename_map[col] = "Open"
            elif low == "high":
                rename_map[col] = "High"
            elif low == "low":
                rename_map[col] = "Low"
            elif low in ("close", "adj close", "adj_close", "price"):
                if "Close" not in rename_map.values():
                    rename_map[col] = "Close"
            elif low in ("volume", "vol"):
                rename_map[col] = "Volume"
            elif low == "ema":
                rename_map[col] = "EMA"
            elif low == "rsi":
                rename_map[col] = "RSI"
            elif low in ("bb_width", "bbwidth", "bollinger_width"):
                rename_map[col] = "BB_Width"
        df = df.rename(columns=rename_map)

        if "Close" not in df.columns:
            raise ValueError(f"Price dataset must contain a 'Close' column. Found: {list(df.columns)}")
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 1.0

        close = df["Close"].astype(float)

        if "EMA" not in df.columns:
            df["EMA"] = close.ewm(span=14, adjust=False).mean()

        if "RSI" not in df.columns:
            delta = close.diff().fillna(0.0)
            gain = delta.clip(lower=0.0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.clip(upper=0.0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        if "BB_Width" not in df.columns:
            rolling_mean = close.rolling(window=20, min_periods=1).mean()
            rolling_std = close.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
            upper_band = rolling_mean + 2.0 * rolling_std
            lower_band = rolling_mean - 2.0 * rolling_std
            df["BB_Width"] = (upper_band - lower_band) / (rolling_mean.abs() + 1e-5)

        return df[PRICE_FEATURE_COLUMNS]

    def load_raw_price_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads raw OHLCV CSV/Parquet file and computes the 8 technical channels."""
        path = Path(file_path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        df = self._parse_date_index(df)
        df = df[~df.index.duplicated(keep="last")]
        return self.compute_technical_indicators(df)

    def load_raw_sentiment_file(
        self,
        file_path: Union[str, Path],
        target_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Loads raw social sentiment dataset from `file_path`.
        Supports:
        1. Pre-computed polarity columns (`Prob_pos, Prob_neg, Prob_neut` or `positive, negative, neutral`)
        2. Raw text comments (`text`, `comment`, `body`, `title`, `sentiment_text`) -> encoded via FinBERT
        3. Single scalar sentiment score (`sentiment`, `score`, `compound`) -> mapped to 3 probabilities.
        """
        path = Path(file_path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        df = self._parse_date_index(df)

        col_lower = {str(c).strip().lower(): c for c in df.columns}

        # Case 1: 3 polarity columns already present
        pos_c = col_lower.get("prob_pos") or col_lower.get("positive") or col_lower.get("pos")
        neg_c = col_lower.get("prob_neg") or col_lower.get("negative") or col_lower.get("neg")
        neu_c = col_lower.get("prob_neut") or col_lower.get("neutral") or col_lower.get("neu")

        if pos_c and neg_c and neu_c:
            pol_df = df[[pos_c, neg_c, neu_c]].astype(float)
            pol_df.columns = SENTIMENT_COLUMNS
            pol_df = pol_df.groupby(pol_df.index).mean()
        else:
            # Case 2: Raw text column -> run FinBERT polarity extractor
            text_col = None
            for cand in ["text", "comment", "body", "sentiment_text", "content", "title", "tweet", "post"]:
                if cand in col_lower:
                    text_col = col_lower[cand]
                    break

            if text_col is not None:
                if target_index is not None:
                    df = df[df.index.isin(target_index)]
                daily_texts = df.groupby(df.index)[text_col].apply(
                    lambda s: " . ".join(str(v) for v in s.iloc[:20] if pd.notna(v))
                )
                probs = self.extract_finbert_sentiment(
                    daily_texts.tolist(),
                    load_pretrained=self.load_pretrained_nlp,
                )
                pol_df = pd.DataFrame(probs, index=daily_texts.index, columns=SENTIMENT_COLUMNS)
            else:
                # Case 3: Single numeric sentiment score in [-1, 1]
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    raise ValueError(
                        f"Sentiment file {path} must contain text or polarity columns. Found: {list(df.columns)}"
                    )
                score = np.clip(df[num_cols[0]].to_numpy(dtype=np.float32), -1.0, 1.0)
                p_pos = np.maximum(score, 0.0) * 0.7 + 0.1
                p_neg = np.maximum(-score, 0.0) * 0.7 + 0.1
                p_neu = np.clip(1.0 - (p_pos + p_neg), 0.0, 1.0)
                pol_df = pd.DataFrame(
                    np.stack([p_pos, p_neg, p_neu], axis=-1),
                    index=df.index,
                    columns=SENTIMENT_COLUMNS,
                ).groupby(level=0).mean()

        # Normalize rows to sum to 1.0
        row_sums = pol_df.to_numpy().sum(axis=-1, keepdims=True)
        pol_df = pd.DataFrame(
            pol_df.to_numpy() / np.maximum(row_sums, 1e-8),
            index=pol_df.index,
            columns=SENTIMENT_COLUMNS,
        )

        if target_index is not None:
            pol_df = pol_df.reindex(target_index)
            # Fill missing sentiment days with neutral [0.0, 0.0, 1.0]
            missing_rows = pol_df["Prob_pos"].isna()
            pol_df.loc[missing_rows, "Prob_pos"] = 0.0
            pol_df.loc[missing_rows, "Prob_neg"] = 0.0
            pol_df.loc[missing_rows, "Prob_neut"] = 1.0

        return pol_df.astype(np.float32)

    def load_raw_news_file(
        self,
        file_path: Union[str, Path],
        target_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Loads raw expert financial news dataset from `file_path` and aligns to `target_index`.
        Supports:
        1. `.npy` or `.pt` array of shape `[T, 256]`
        2. CSV/Parquet with raw text headlines (`headline`, `title`, `news`, `text`, `summary`)
           -> encoded via DistilBERT into R^{T x 256}
        3. CSV/Parquet with pre-extracted numeric embedding columns.
        """
        path = Path(file_path)
        T = len(target_index)

        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
            if arr.shape[0] == T:
                if arr.shape[1] < self.news_embed_dim:
                    pad = np.zeros((T, self.news_embed_dim - arr.shape[1]), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=1)
                elif arr.shape[1] > self.news_embed_dim:
                    arr = arr[:, : self.news_embed_dim]
                return arr
            raise ValueError(
                f"News .npy array shape {arr.shape} does not match expected ({T}, {self.news_embed_dim})."
            )

        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        df = self._parse_date_index(df)
        col_lower = {str(c).strip().lower(): c for c in df.columns}

        text_col = None
        for cand in [
            "combined_text",
            "headline",
            "headlines",
            "title",
            "news",
            "text",
            "article",
            "summary",
            "content",
        ]:
            if cand in col_lower:
                text_col = col_lower[cand]
                break

        if text_col is not None:
            # Concatenate multiple headlines on the same date before DistilBERT extraction
            daily_texts = df.groupby(df.index)[text_col].apply(
                lambda s: " . ".join(str(v) for v in s if pd.notna(v))
            )
            embs = self.extract_distilbert_news_embeddings(
                daily_texts.tolist(),
                d_embed=self.news_embed_dim,
                load_pretrained=self.load_pretrained_nlp,
            )
            emb_df = pd.DataFrame(embs, index=daily_texts.index)
        else:
            num_df = df.select_dtypes(include=[np.number]).groupby(level=0).mean()
            vals = num_df.to_numpy(dtype=np.float32)
            if vals.shape[1] < self.news_embed_dim:
                pad = np.zeros((vals.shape[0], self.news_embed_dim - vals.shape[1]), dtype=np.float32)
                vals = np.concatenate([vals, pad], axis=1)
            elif vals.shape[1] > self.news_embed_dim:
                vals = vals[:, : self.news_embed_dim]
            emb_df = pd.DataFrame(vals, index=num_df.index)

        emb_df = emb_df.reindex(target_index).ffill().bfill().fillna(0.0)
        return emb_df.to_numpy(dtype=np.float32)

    def load_raw_macro_file(
        self,
        file_path: Union[str, Path],
        target_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Loads raw macroeconomic inflation dataset from `file_path` and aligns to `target_index`.
        Preserves the structural monthly reporting schedule (only 1 non-NaN observation
        per month / every 30 trading steps; intermediate daily rows remain NaN in the
        returned DataFrame so `align_streams` can compute exact latency delta_T before forward-filling).
        """
        path = Path(file_path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        df = self._parse_date_index(df)
        df = df[~df.index.duplicated(keep="last")]

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            # Convert string numeric columns that may contain '.' or blanks (like FRED CSVs)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            raise ValueError(f"Macro file {path} contains no numeric inflation columns.")

        selected_cols = num_cols[: self.macro_dim]
        macro_sub = df[selected_cols].copy()

        # Map each valid macro release date to the nearest trading date on or after in target_index
        aligned_macro = pd.DataFrame(
            np.nan,
            index=target_index,
            columns=[f"CPI_YoY_{i}" for i in range(len(selected_cols))],
            dtype=np.float32,
        )

        valid_macro = macro_sub.dropna(how="all")
        if len(valid_macro) <= (len(target_index) // 15):
            # Monthly/sparse release table: snap each release to its corresponding trading session
            for rel_date, row_vals in valid_macro.iterrows():
                pos = target_index.searchsorted(rel_date)
                if pos < len(target_index):
                    aligned_macro.iloc[pos] = row_vals.to_numpy(dtype=np.float32)
        else:
            # Daily table: reindex directly; if all rows were pre-filled without NaNs,
            # enforce the 30-step structural release schedule (FR1 / Phase 2)
            reindexed = macro_sub.reindex(target_index)
            if not reindexed.iloc[:, 0].isna().any():
                mask_non_release = (np.arange(len(target_index)) % self.macro_period_steps) != 0
                reindexed.iloc[mask_non_release, :] = np.nan
            aligned_macro.iloc[:, :] = reindexed.to_numpy(dtype=np.float32)

        return aligned_macro

    @staticmethod
    def _resolve_modality_file(
        data_path: Path,
        asset_name: str,
        candidates: List[str],
    ) -> Optional[Path]:
        """Finds the first matching file in `data_path` for any arbitrary `asset_name`."""
        asset_slug = asset_name.strip().lower().replace(" ", "_").replace("-", "_")
        slug_aliases = [asset_slug]
        if "_" in asset_slug:
            parts = [p for p in asset_slug.split("_") if p]
            if parts:
                for token in (parts[-1], parts[0], "".join(parts)):
                    if token not in slug_aliases:
                        slug_aliases.append(token)
        if asset_slug in ("brent_crude_oil", "crude_oil", "oil"):
            for alias in ("brent_crude_oil", "crude_oil", "oil"):
                if alias not in slug_aliases:
                    slug_aliases.append(alias)

        search_names: List[str] = []
        for cand in candidates:
            for slug in slug_aliases:
                search_names.append(f"{slug}_{cand}")
            search_names.append(cand)

        search_dirs = [data_path]
        if data_path.resolve() == DEFAULT_DATA_DIR.resolve():
            search_dirs.append(data_path.resolve().parent)

        for d in search_dirs:
            if not d.exists():
                continue
            existing_lower = {f.name.lower(): f for f in d.iterdir() if f.is_file()}
            for name in search_names:
                matched = existing_lower.get(name.lower())
                if matched is not None:
                    return matched
        return None

    @classmethod
    def discover_assets_in_folder(
        cls,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    ) -> List[str]:
        """
        Automatically discovers all assets present in `data_dir` by scanning for price
        time-series files (`<asset>_price_data.csv`, `<asset>_price.csv`, `<asset>_ohlcv.csv`,
        or generic `price.csv` / `price_data.csv`).
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")

        price_suffixes = (
            "_price_data.csv",
            "_price.csv",
            "_ohlcv.csv",
            "_price_data.parquet",
            "_price.parquet",
            "_ohlcv.parquet",
        )
        generic_price_names = {
            "price_data.csv",
            "price.csv",
            "ohlcv.csv",
            "price_data.parquet",
            "price.parquet",
            "ohlcv.parquet",
        }

        discovered_slugs: List[str] = []
        has_generic_price = False

        for f in sorted(data_path.iterdir()):
            if not f.is_file():
                continue
            fname_low = f.name.lower()
            if fname_low in generic_price_names:
                has_generic_price = True
                continue
            for suf in price_suffixes:
                if fname_low.endswith(suf):
                    prefix = f.name[: len(f.name) - len(suf)].strip("_- ")
                    if prefix and prefix.lower() not in [s.lower() for s in discovered_slugs]:
                        discovered_slugs.append(prefix)
                    break

        if not discovered_slugs and has_generic_price:
            discovered_slugs.append("Default")

        return discovered_slugs

    def align_streams(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        news_embeddings: np.ndarray,
        macro_df: pd.DataFrame,
        asset_name: str = "Gold",
    ) -> AlignedDatasetBundle:
        """
        Aligns the 4 heterogeneous streams into rolling windowed tensors for each day t:
        - X_OHLCV^{(t)} in R^{h x 8}
        - X_sentiment^{(t)} in R^{h x 3} (FinBERT polarities)
        - X_Expert^{(t)} in R^{d x 256} (DistilBERT contextual representations)
        - X_macro^{(t)} in R^{k} (forward-filled from monthly release)
        - delta_T^{(t)} in R^{4} (temporal ingestion latency)
        - S_state^{(t)} in R^{V} (market state volatility indicators)
        - a_close^{(t-h, t)} in R^{1} (rolling h-step close average)
        - y_target^{(t+1)} in R^{1} (next-day close price)
        """
        if not set(PRICE_FEATURE_COLUMNS).issubset(price_df.columns):
            price_df = self.compute_technical_indicators(price_df)

        common_index = price_df.index.intersection(sentiment_df.index).intersection(macro_df.index)
        price_df = price_df.loc[common_index]
        sentiment_df = sentiment_df.loc[common_index]
        macro_df = macro_df.loc[common_index]

        T = len(common_index)
        if news_embeddings.shape[0] != T:
            raise ValueError(
                f"news_embeddings length ({news_embeddings.shape[0]}) must match aligned index ({T})."
            )

        raw_macro_vec = macro_df.iloc[:, 0].to_numpy(dtype=np.float32)

        # Calculate days since last macro publication (delta_T for modality 3)
        macro_latency = np.zeros(T, dtype=np.float32)
        days_since_release = 0.0
        for idx in range(T):
            if not np.isnan(raw_macro_vec[idx]):
                days_since_release = 0.0
            else:
                days_since_release += 1.0
            macro_latency[idx] = days_since_release / float(self.macro_period_steps)

        macro_ffill = macro_df.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        if macro_ffill.ndim == 1:
            macro_ffill = macro_ffill[:, None]

        price_arr = price_df[PRICE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        sent_arr = sentiment_df[SENTIMENT_COLUMNS].to_numpy(dtype=np.float32)
        news_arr = np.asarray(news_embeddings, dtype=np.float32)
        close_series = price_df["Close"].to_numpy(dtype=np.float32)

        # Market State Volatility Vector S(t) = [S_1(t), S_2(t), S_3(t)] in R^3:
        # S_1(t): Rolling 10-day volatility of log-returns r(t) = ln(P_close(t) / P_close(t-1))
        # S_2(t): Centred Relative Strength Index (RSI_14 - 50) / 50
        # S_3(t): Scaled Bollinger Band Width 10 * (B_upper - B_lower) / (|mu_20(t)| + 1e-5)
        close_s = pd.Series(close_series, dtype=np.float64)
        log_returns = np.log(close_s.clip(lower=1e-5) / close_s.shift(1).clip(lower=1e-5)).fillna(0.0)
        vol_10d = log_returns.rolling(10, min_periods=1).std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
        rsi_norm = (price_df["RSI"].to_numpy(dtype=np.float32) - 50.0) / 50.0

        rolling_mean_20 = close_s.rolling(window=20, min_periods=1).mean()
        rolling_std_20 = close_s.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
        bb_width = (
            (4.0 * rolling_std_20) / (rolling_mean_20.abs() + 1e-5)
        ).to_numpy(dtype=np.float32)
        market_state_all = np.stack([vol_10d * 100.0, rsi_norm, bb_width * 10.0], axis=-1)

        start_idx = max(self.lookback_h, self.news_delay_d)
        num_samples = T - start_idx
        if num_samples <= 0:
            raise ValueError(
                f"Insufficient rows ({T}) for lookback_h={self.lookback_h} and next-day target."
            )

        X_OHLCV_list = []
        X_sent_list = []
        X_news_list = []
        X_macro_list = []
        delta_T_list = []
        S_state_list = []
        a_close_list = []
        y_target_list = []
        P_current_list = []
        valid_dates = []

        for t in range(start_idx - 1, T - 1):
            p_win = price_arr[t - self.lookback_h + 1 : t + 1]
            s_win = sent_arr[t - self.lookback_h + 1 : t + 1]
            n_win = news_arr[t - self.news_delay_d + 1 : t + 1]
            m_val = macro_ffill[t]

            a_close_val = np.mean(close_series[t - self.lookback_h + 1 : t + 1], keepdims=True)
            dt_vec = np.array([0.0, 0.02, 0.05, macro_latency[t]], dtype=np.float32)

            X_OHLCV_list.append(p_win)
            X_sent_list.append(s_win)
            X_news_list.append(n_win)
            X_macro_list.append(m_val)
            delta_T_list.append(dt_vec)
            S_state_list.append(market_state_all[t, : self.v_indices])
            a_close_list.append(a_close_val.astype(np.float32))
            P_current_list.append(np.array([close_series[t]], dtype=np.float32))
            y_target_list.append(np.array([close_series[t + 1]], dtype=np.float32))
            valid_dates.append(common_index[t + 1])

        return AlignedDatasetBundle(
            X_OHLCV=np.stack(X_OHLCV_list, axis=0),
            X_sentiment=np.stack(X_sent_list, axis=0),
            X_Expert=np.stack(X_news_list, axis=0),
            X_macro=np.stack(X_macro_list, axis=0),
            raw_macro_series=raw_macro_vec,
            delta_T=np.stack(delta_T_list, axis=0),
            S_state=np.stack(S_state_list, axis=0),
            a_close=np.stack(a_close_list, axis=0),
            y_target=np.stack(y_target_list, axis=0),
            P_current=np.stack(P_current_list, axis=0),
            timestamps=pd.DatetimeIndex(valid_dates),
            asset_name=asset_name,
        )

    def load_from_data_folder(
        self,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        asset_name: str = "Gold",
        price_file: Optional[Union[str, Path]] = None,
        sentiment_file: Optional[Union[str, Path]] = None,
        news_file: Optional[Union[str, Path]] = None,
        macro_file: Optional[Union[str, Path]] = None,
        num_days: Optional[int] = None,
    ) -> AlignedDatasetBundle:
        """
        Strictly loads the 4 raw datasets (`price`, `sentiment`, `news`, `macro`)
        from `data_dir` (defaulting to `<project_root>/data`).
        Raises `FileNotFoundError` if `data_dir` or any of the 4 modality files is missing.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory '{data_path}' does not exist. "
                "Please place your raw datasets (price.csv, sentiment.csv, news.csv, macro.csv) in the data/ folder."
            )

        p_path = Path(price_file) if price_file else self._resolve_modality_file(
            data_path,
            asset_name,
            ["price_data.csv", "price.csv", "ohlcv.csv", "price.parquet", "ohlcv.parquet"],
        )
        s_path = Path(sentiment_file) if sentiment_file else self._resolve_modality_file(
            data_path,
            asset_name,
            ["sentiments.csv", "sentiment.csv", "social_sentiment.csv", "sentiment.parquet"],
        )
        n_path = Path(news_file) if news_file else self._resolve_modality_file(
            data_path,
            asset_name,
            ["news_data.csv", "news.csv", "expert_news.csv", "news.parquet", "news.npy"],
        )
        m_path = Path(macro_file) if macro_file else self._resolve_modality_file(
            data_path,
            asset_name,
            ["inflation_data.csv", "macro.csv", "inflation.csv", "cpi.csv", "macro.parquet"],
        )

        missing_modalities = []
        if not p_path or not p_path.exists():
            missing_modalities.append("price (e.g., price.csv)")
        if not s_path or not s_path.exists():
            missing_modalities.append("sentiment (e.g., sentiment.csv)")
        if not n_path or not n_path.exists():
            missing_modalities.append("news (e.g., news.csv)")
        if not m_path or not m_path.exists():
            missing_modalities.append("macro (e.g., macro.csv)")

        if missing_modalities:
            raise FileNotFoundError(
                f"Missing required raw dataset file(s) for asset '{asset_name}' in '{data_path}': "
                f"{', '.join(missing_modalities)}. Synthetic data generation is disabled."
            )

        price_df = self.load_raw_price_file(p_path)
        if num_days is not None:
            price_df = price_df.iloc[:num_days]
        target_index = price_df.index
        sent_df = self.load_raw_sentiment_file(s_path, target_index=target_index)
        news_emb = self.load_raw_news_file(n_path, target_index=target_index)
        macro_df = self.load_raw_macro_file(m_path, target_index=target_index)

        return self.align_streams(
            price_df=price_df,
            sentiment_df=sent_df,
            news_embeddings=news_emb,
            macro_df=macro_df,
            asset_name=asset_name,
        )

    def build_aligned_dataset(
        self,
        asset_name: str = "Gold",
        num_days: Optional[int] = None,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        price_file: Optional[Union[str, Path]] = None,
        sentiment_file: Optional[Union[str, Path]] = None,
        news_file: Optional[Union[str, Path]] = None,
        macro_file: Optional[Union[str, Path]] = None,
    ) -> AlignedDatasetBundle:
        """
        Strictly loads and aligns the 4 raw historical datasets from `data_dir`
        (default: `<workspace>/data`). Synthetic data generation is completely disabled.
        """
        if data_dir is None:
            raise ValueError("`data_dir` cannot be None; raw datasets must be loaded from the data/ folder.")

        return self.load_from_data_folder(
            data_dir=data_dir,
            asset_name=asset_name,
            price_file=price_file,
            sentiment_file=sentiment_file,
            news_file=news_file,
            macro_file=macro_file,
            num_days=num_days,
        )
