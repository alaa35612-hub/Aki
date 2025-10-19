# -*- coding: utf-8 -*-
"""
SMC (Pine-Equivalent 1:1) + Binance Futures Scanner + Telegram Notifier

- Inside/Mother: STRICT (> و < فقط) — شمعة واحدة (بدون سلسلة)
- H/L update: >= و <= كما في Pine
- OBS/OBD من [b-2] + Mother[b-2] عند isb[2] (pivot [2-1] فقط) — (تم إصلاح off-by-one)
- BOS/CHoCh (close-only) + EXT(NEAREST) + IDM take (WICK/CLOSE) + أول ملامسة IDM
- DigitalBookkeeping (بديل رقمي للرسومات)
- لا رسومات إطلاقًا — مخرجات رقمية فقط

- تحسينات تشغيل:
  --symbol       : اختيار رمز واحد
  --recent N     : طباعة تنبيهات آخر N شموع (افتراضي 1)
  --verbose      : ملخّص حتى بدون تنبيهات
  --drop-last    : تجاهل آخر شمعة (مفيد لمضاهاة Pine على الشموع المؤكدة فقط)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Literal
from collections import deque
import numpy as np
import pandas as pd
import os
import sys

# ---------------- Optional deps ----------------
try:
    from binance.client import Client
except Exception:
    Client = None

try:
    from telegram import Bot
except Exception:
    Bot = None

# ======== مفاتيح داخلية (يمكن تركها فارغة لاستخدام متغيّرات البيئة بدلاً منها) ========
# ملاحظة أمنيّة: تخزين المفاتيح في النص الصريح داخل الملف غير مستحسن في بيئات مشاركة.
# استخدم هذا فقط للاستخدام المحلي كما ذكرت — ولا تضع الملف في مستودع عام.
API_KEYS = {
    "BINANCE_API_KEY":     "joGy8hFXVGvuWFiVXoqRncscGZxlgsARycdT yzl2kJjR2675bGMhHuonO1AyOQSX",  # ضع هنا مفتاح Binance API (أو اتركه فارغًا لاستخدام متغير البيئة)
    "BINANCE_API_SECRET":  "utDzvgS7sAdP4W82hxyksXk1VhrLYeyPmYq QWV6MRGaNgakx3FiggRASYxaaqnbf",  # ضع هنا السر الخاص بـ Binance (أو اتركه فارغًا لاستخدام متغير البيئة)
    "TELEGRAM_BOT_TOKEN":  "",  # ضع هنا توكن البوت إن رغبت بالإشعارات
    "TELEGRAM_CHAT_ID":    "",  # ضع هنا chat id إن رغبت بالإشعارات
}
def _get_secret(name: str, default: str = "") -> str:
    """يرجع المفتاح الداخلي إذا كان مضبوطًا، وإلا يرجع من متغيرات البيئة، وإلا default."""
    val = API_KEYS.get(name, "") or os.getenv(name, "") or default
    return str(val).strip()

# =========================
# الإعدادات العامة
# =========================
@dataclass
class Settings:
    eps: float = 1e-10
    enable_alert_bos: bool = True
    enable_alert_choch: bool = True
    enable_alert_idm_touch: bool = True
    poi_type_mother_bar: bool = True
    mitigation_mode: str = "WICK"   # "WICK" (ملامسة الذيل) أو "CLOSE" (إغلاق)
    merge_ratio: float = 0.10
    tg_enable: bool = False
    tg_title_prefix: str = "SMC Alert"


# =========================
# Digital Bookkeeping (بديل رقمي للرسومات)
# =========================
@dataclass
class DigitalBookkeeping:
    arrLastH: List[float] = field(default_factory=list)
    arrLastHBar: List[int] = field(default_factory=list)
    arrLastL: List[float] = field(default_factory=list)
    arrLastLBar: List[int] = field(default_factory=list)

    arrIdmHigh: List[float] = field(default_factory=list)
    arrIdmHBar: List[int] = field(default_factory=list)
    arrIdmLow: List[float] = field(default_factory=list)
    arrIdmLBar: List[int] = field(default_factory=list)

    # “رسومات” رقمية بديلة — نُراعي عدد الحذف كما في Pine
    arrBCLabel: List[int] = field(default_factory=list)
    arrBCLine: List[int] = field(default_factory=list)
    arrHLLabel: List[int] = field(default_factory=list)
    arrHLCircle: List[int] = field(default_factory=list)
    arrIdmLabel: List[int] = field(default_factory=list)
    arrIdmLine: List[int] = field(default_factory=list)

    # push
    def updateLastH(self, val: float, bar_time: int):
        self.arrLastH.append(val); self.arrLastHBar.append(bar_time)
    def updateLastL(self, val: float, bar_time: int):
        self.arrLastL.append(val); self.arrLastLBar.append(bar_time)
    def updateIdmHigh(self, val: float, bar_time: int):
        self.arrIdmHigh.append(val); self.arrIdmHBar.append(bar_time)
    def updateIdmLow(self, val: float, bar_time: int):
        self.arrIdmLow.append(val); self.arrIdmLBar.append(bar_time)

    def pushBCLabel(self, code: int): self.arrBCLabel.append(code)
    def pushBCLine(self, code: int):  self.arrBCLine.append(code)
    def pushHLLabel(self, code: int): self.arrHLLabel.append(code)
    def pushHLCircle(self, code: int):self.arrHLCircle.append(code)
    def pushIdmLabel(self, code: int):self.arrIdmLabel.append(code)
    def pushIdmLine(self, code: int): self.arrIdmLine.append(code)

    # pop helpers
    def _pop_safe(self, arr: List):
        if arr: arr.pop()
    def _pop_n(self, arr: List, n: int):
        for _ in range(min(n, len(arr))): arr.pop()

    # مطابق لعدد الحذف في Pine
    def fixStrcAfterBos(self):
        self._pop_safe(self.arrBCLabel); self._pop_safe(self.arrBCLine)
        self._pop_safe(self.arrIdmLabel); self._pop_safe(self.arrIdmLine)
        self._pop_n(self.arrHLLabel, 2); self._pop_n(self.arrHLCircle, 2)

    def fixStrcAfterChoch(self):
        self._pop_n(self.arrBCLabel, 2); self._pop_n(self.arrBCLine, 2)
        self._pop_n(self.arrHLLabel, 3); self._pop_n(self.arrHLCircle, 3)
        self._pop_n(self.arrIdmLabel, 2); self._pop_n(self.arrIdmLine, 2)


# =========================
# Zone Struct
# =========================
ZoneKind = Literal["SUPPLY", "DEMAND"]
ZoneState = Literal["FRESH", "ACTIVATED", "MITIGATED", "IDM_AND_MIT"]
ZoneLabel = Literal["", "EXT OB", "Hist EXT OB", "IDM OB", "Hist IDM OB"]

@dataclass
class Zone:
    left_index: int
    right_index: int
    top: float
    bottom: float
    kind: ZoneKind
    label: ZoneLabel = ""
    state: ZoneState = "FRESH"


# =========================
# Telegram Notifier (اختياري)
# =========================
class TelegramNotifier:
    def __init__(self):
        token = _get_secret("TELEGRAM_BOT_TOKEN")
        chat = _get_secret("TELEGRAM_CHAT_ID")
        self.enabled = bool(Bot and token and chat)
        self._bot = Bot(token=token) if self.enabled else None
        self.chat_id = chat

    def send(self, text: str):
        if self.enabled and self._bot:
            try:
                self._bot.send_message(chat_id=self.chat_id, text=text)
            except Exception as e:
                print(f"[TELEGRAM ERROR] {e}\n{text}")
        else:
            print("[ALERT] " + text)


# =========================
# SMCIndicator (Pine-equivalent 1:1)
# =========================
class SMCIndicator:
    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or Settings()
        self.bk = DigitalBookkeeping()

        # Anchors عامة (H/L) — تحدّث بـ >= / <=
        self.H = -np.inf; self.L = np.inf
        self.HBar = -1;   self.LBar = -1

        # Anchors حدثيّة — تتحرّك على أحداث (BOS/CHoCh/IDM take)
        self.lastH = -np.inf; self.lastL = np.inf
        self.lastHBar = -1;   self.lastLBar = -1

        # IDM
        self.idmHigh = np.nan; self.idmLow = np.nan
        self.idmHBar = -1;     self.idmLBar = -1

        # PU snapshot
        self.puHigh = -np.inf; self.puLow = np.inf
        self.puHBar = -1;      self.puLBar = -1

        # حالات الهيكل
        self.findIDM = False
        self.isCocUp = False;  self.isCocDn = False
        self.isBosUp = False;  self.isBosDn = False
        self.isPrevBos = False
        self.has_bos = False

        # Mother/Inside Bar (STRICT) — شمعة واحدة
        self.motherHigh = -np.inf
        self.motherLow  = np.inf
        self.motherBar  = -1
        self.isb_hist = deque(maxlen=5)
        self.motherHighHist: List[float] = []
        self.motherLowHist:  List[float] = []
        self.motherBarHist:  List[int] = []

        # مناطق
        self.zones: List[Zone] = []

        # نواتج
        self.alerts: List[Dict] = []
        self._bos_up: List[bool] = []
        self._bos_dn: List[bool] = []
        self._choch_up: List[bool] = []
        self._choch_dn: List[bool] = []
        self._idm_touch_supply: List[bool] = []
        self._idm_touch_demand: List[bool] = []
        self._idm_touch_seen_this_bar = set()

    # ---------- Helpers ----------
    @staticmethod
    def _is_green(op: float, cl: float) -> bool:
        return cl >= op

    def _isb_ago(self, bars_back: int) -> bool:
        if len(self.isb_hist) <= bars_back:
            return False
        return bool(list(self.isb_hist)[-1 - bars_back])

    # ---------- IDM buffers update (كما في Pine) ----------
    def _update_idm_buffers(self, hi: float, lo: float, op: float, cl: float, b: int):
        is_green = self._is_green(op, cl)
        in_uptrend = self.isCocUp or self.isBosUp
        in_downtrend = self.isCocDn or self.isBosDn
        in_range = not (in_uptrend or in_downtrend)

        # PU snapshot
        self.puHigh, self.puLow = hi, lo
        self.puHBar, self.puLBar = b, b

        if in_range:
            if (lo < self.L) and is_green:
                self.bk.updateIdmHigh(self.puHigh, self.puHBar)
            if (hi > self.H) and (not is_green):
                self.bk.updateIdmLow(self.puLow, self.puLBar)
        elif in_uptrend:
            if hi > self.H:
                self.bk.updateIdmLow(self.puLow, self.puLBar)
        elif in_downtrend:
            if lo < self.L:
                self.bk.updateIdmHigh(self.puHigh, self.puHBar)

    # ---------- Mother / Inside Bar (STRICT) ----------
    def _update_mother_bar(self, hi: float, lo: float, b: int):
        # isb = motherHigh > high and motherLow < low (STRICT: > / < فقط)
        if self.motherBar == -1:
            self.motherHigh = hi; self.motherLow = lo; self.motherBar = b
            self.isb_hist.append(False)
        else:
            is_inside = (self.motherHigh > hi) and (self.motherLow < lo)
            self.isb_hist.append(is_inside)
            if not is_inside:
                self.motherHigh = hi; self.motherLow = lo; self.motherBar = b
        self.motherHighHist.append(self.motherHigh)
        self.motherLowHist.append(self.motherLow)
        self.motherBarHist.append(self.motherBar)

    # ---------- OBS/OBD من [b-2] + Mother[b-2] عند isb[2] ----------
    def _build_obs_obd_from_b2(self, b: int, df: pd.DataFrame):
        if b < 2:
            return
        idx2 = b - 2
        hi2 = float(df.high.iloc[idx2])
        lo2 = float(df.low.iloc[idx2])

        use_mother = self.cfg.poi_type_mother_bar and self._isb_ago(2)
        if use_mother and len(self.motherHighHist) >= 3:
            # إصلاح off-by-one: Mother[b-2] = history[-3] عند معالجة البار b
            box_top = self.motherHighHist[-3]
            box_bot = self.motherLowHist[-3]
            left_ix = self.motherBarHist[-3]
        else:
            box_top, box_bot = hi2, lo2
            left_ix = idx2

        # pivot [2-1] فقط
        prev_hi = float(df.high.iloc[idx2 - 1]) if idx2 - 1 >= 0 else -np.inf
        prev_lo = float(df.low .iloc[idx2 - 1]) if idx2 - 1 >= 0 else  np.inf

        # قمة محلية ⇒ SUPPLY
        if hi2 > prev_hi:
            self._merge_or_add_zone(Zone(left_ix, b, box_top, box_bot, "SUPPLY"))
        # قاع محلي ⇒ DEMAND
        if lo2 < prev_lo:
            self._merge_or_add_zone(Zone(left_ix, b, box_top, box_bot, "DEMAND"))

    def _merge_or_add_zone(self, candidate: Zone):
        same_kind = [z for z in reversed(self.zones) if z.kind == candidate.kind]
        if not same_kind:
            self.zones.append(candidate); return
        last = same_kind[0]
        # overlap ratio
        inter_top = min(candidate.top, last.top)
        inter_bot = max(candidate.bottom, last.bottom)
        if inter_top > inter_bot:
            inter = inter_top - inter_bot
            base = max(abs(candidate.top - candidate.bottom), abs(last.top - last.bottom), 1e-12)
            ratio = inter / base
        else:
            ratio = 0.0
        included = (candidate.top >= last.top) and (candidate.bottom <= last.bottom)
        if ratio >= self.cfg.merge_ratio or included:
            last.top = max(last.top, candidate.top)
            last.bottom = min(last.bottom, candidate.bottom)
            last.left_index = min(last.left_index, candidate.left_index)
            last.right_index = max(last.right_index, candidate.right_index)
        else:
            self.zones.append(candidate)

    # ---------- EXT = NEAREST على الكسر ----------
    def _select_ext_on_break(self, trend_up: bool):
        y = self.lastH if trend_up else self.lastL
        anchor = self.lastL if trend_up else self.lastH
        pool: List[Zone] = []
        for z in self.zones:
            if z.state in ("MITIGATED", "IDM_AND_MIT"):
                continue
            if trend_up and z.kind == "DEMAND":
                if (z.top <= y + self.cfg.eps) and (z.bottom >= anchor - self.cfg.eps):
                    pool.append(z)
            if (not trend_up) and z.kind == "SUPPLY":
                if (z.bottom >= y - self.cfg.eps) and (z.top <= anchor + self.cfg.eps):
                    pool.append(z)
        if not pool:
            return
        chosen = sorted(pool, key=lambda z: abs((z.top if trend_up else z.bottom) - y))[0]
        for z in self.zones:
            if z.kind == chosen.kind and z.label == "EXT OB":
                z.label = "Hist EXT OB"
        chosen.label = "EXT OB"
        chosen.state = "ACTIVATED"

    # ---------- تفعيل/أخذ IDM + أول ملامسة ----------
    def _activate_idm_from_candidates(self, trend_up: bool):
        y = self.idmLow if trend_up else self.idmHigh
        anchor = self.lastL if trend_up else self.lastH
        if not np.isfinite(y):
            return
        candidates: List[Tuple[int, Zone, float]] = []
        for idx, z in enumerate(self.zones):
            if z.state != "FRESH":
                continue
            if trend_up and z.kind == "DEMAND":
                if (z.top <= y + self.cfg.eps) and (z.bottom >= anchor - self.cfg.eps):
                    candidates.append((idx, z, -z.top))   # الأعلى أولًا
            if (not trend_up) and z.kind == "SUPPLY":
                if (z.bottom >= y - self.cfg.eps) and (z.top <= anchor + self.cfg.eps):
                    candidates.append((idx, z, z.bottom)) # الأدنى أولًا
        if not candidates:
            return
        _, chosen, _ = sorted(candidates, key=lambda t: t[2])[0]
        for z2 in self.zones:
            if z2.label == "IDM OB":
                z2.label = "Hist IDM OB"
        chosen.label = "IDM OB"
        chosen.state = "ACTIVATED"

    def _idm_take_unlock(self, trend_up: bool, hi: float, lo: float, cl: float, i: int):
        if trend_up:
            take = (lo < self.idmLow - self.cfg.eps) if self.cfg.mitigation_mode == "WICK" \
                   else (cl < self.idmLow - self.cfg.eps)
            if take:
                self.findIDM = False; self.isBosUp = False
                self._activate_idm_from_candidates(True)
                # مراسي حدثية بعد الأخذ
                self.lastH, self.lastHBar = self.H, self.HBar
        else:
            take = (hi > self.idmHigh + self.cfg.eps) if self.cfg.mitigation_mode == "WICK" \
                   else (cl > self.idmHigh + self.cfg.eps)
            if take:
                self.findIDM = False; self.isBosDn = False
                self._activate_idm_from_candidates(False)
                self.lastL, self.lastLBar = self.L, self.LBar

    def _first_touch_idm(self, i: int, df: pd.DataFrame):
        if not self.cfg.enable_alert_idm_touch or i < 1:
            return (False, False)
        self._idm_touch_seen_this_bar.clear()
        prev_high = float(df.high.iloc[i-1]); prev_low  = float(df.low.iloc[i-1])
        hi = float(df.high.iloc[i]);          lo = float(df.low.iloc[i])
        touch_s = touch_d = False
        for z in self.zones:
            if not (z.label == "IDM OB" and z.state == "ACTIVATED"):
                continue
            key = (z.kind, z.left_index, z.right_index)
            if key in self._idm_touch_seen_this_bar:
                continue
            if z.kind == "SUPPLY":
                is_new = (hi >= z.bottom - self.cfg.eps) and (prev_high < z.bottom - self.cfg.eps)
                if is_new:
                    touch_s = True; self._idm_touch_seen_this_bar.add(key)
                    self.alerts.append({"bar": i, "type": "IDM Supply Touch", "price": z.bottom, "box": [z.bottom, z.top]})
            else:
                is_new = (lo <= z.top + self.cfg.eps) and (prev_low > z.top + self.cfg.eps)
                if is_new:
                    touch_d = True; self._idm_touch_seen_this_bar.add(key)
                    self.alerts.append({"bar": i, "type": "IDM Demand Touch", "price": z.top, "box": [z.bottom, z.top]})
        return (touch_s, touch_d)

    # ---------- خريطة الهيكل (BOS/CHoCh) — مع تمهيد الاتجاه لأول BOS ----------
    def _structure_mapping(self, hi: float, lo: float, cl: float, i: int):
        bos_up = bos_dn = choch_up = choch_dn = False

        # CHoCh Up — بوابتها has_bos
        if self.isCocDn and (not self.findIDM) and self.has_bos and (cl > self.lastH + self.cfg.eps):
            self._select_ext_on_break(True)
            self.findIDM = True
            self.isBosUp = self.isCocUp = True
            self.isBosDn = self.isCocDn = False
            self.isPrevBos = False
            self.has_bos = False
            self.lastL, self.lastLBar = self.L, self.LBar
            choch_up = True
            if self.cfg.enable_alert_choch:
                self.alerts.append({"bar": i, "type": "CHoCh Up", "price": self.lastH})

        # CHoCh Down
        if self.isCocUp and (not self.findIDM) and self.has_bos and (cl < self.lastL - self.cfg.eps):
            self._select_ext_on_break(False)
            self.findIDM = True
            self.isBosDn = self.isCocDn = True
            self.isBosUp = self.isCocUp = False
            self.isPrevBos = False
            self.has_bos = False
            self.lastH, self.lastHBar = self.H, self.HBar
            choch_dn = True
            if self.cfg.enable_alert_choch:
                self.alerts.append({"bar": i, "type": "CHoCh Down", "price": self.lastL})

        # BOS Up — أول BOS بدون isCocUp لتمهيد الاتجاه
        if (not self.findIDM) and (not self.isBosUp) and (cl > self.lastH + self.cfg.eps):
            self.findIDM = True
            self.isBosUp = self.isCocUp = True
            self.isBosDn = self.isCocDn = False
            self.isPrevBos = True
            self.has_bos = True
            self.lastL, self.lastLBar = self.L, self.LBar
            bos_up = True
            self._select_ext_on_break(True)
            if self.cfg.enable_alert_bos:
                self.alerts.append({"bar": i, "type": "BOS Up", "price": self.lastH})

        # BOS Down — أول BOS بدون isCocDn لتمهيد الاتجاه
        if (not self.findIDM) and (not self.isBosDn) and (cl < self.lastL - self.cfg.eps):
            self.findIDM = True
            self.isBosUp = self.isCocUp = False
            self.isBosDn = self.isCocDn = True
            self.isPrevBos = True
            self.has_bos = True
            self.lastH, self.lastHBar = self.H, self.HBar
            bos_dn = True
            self._select_ext_on_break(False)
            if self.cfg.enable_alert_bos:
                self.alerts.append({"bar": i, "type": "BOS Down", "price": self.lastL})

        # IDM take (WICK/CLOSE)
        if self.findIDM and self.isCocUp:
            self._idm_take_unlock(True, hi, lo, cl, i)
        if self.findIDM and self.isCocDn:
            self._idm_take_unlock(False, hi, lo, cl, i)

        return bos_up, bos_dn, choch_up, choch_dn

    # ---------- التشغيل ----------
    def run(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # تحضير الداتا (تأمين الأنواع)
        for col in ['open','high','low','close','volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['open','high','low','close'])

        n = len(df)
        self._bos_up = [False]*n; self._bos_dn = [False]*n
        self._choch_up = [False]*n; self._choch_dn = [False]*n
        self._idm_touch_supply = [False]*n; self._idm_touch_demand = [False]*n

        # تهيئة آمنة
        if n > 0:
            first_hi = float(df['high'].iloc[0])
            first_lo = float(df['low'].iloc[0])
            self.H = first_hi; self.HBar = 0
            self.L = first_lo; self.LBar = 0
            self.lastH = first_hi; self.lastHBar = 0
            self.lastL = first_lo; self.lastLBar = 0

        for i in range(n):
            op = float(df['open'].iloc[i]); hi = float(df['high'].iloc[i])
            lo = float(df['low'].iloc[i]);  cl = float(df['close'].iloc[i])

            # 1) IDM buffers
            self._update_idm_buffers(hi,lo,op,cl,i)

            # 2) Mother / Inside Bar (STRICT)
            self._update_mother_bar(hi,lo,i)

            # 3) تحديث H/L العامة (>= / <=) + قراءة آخر IDM عند التحديث
            if hi >= self.H:
                self.H, self.HBar = hi, i
                if self.bk.arrIdmLow:
                    self.idmLow  = self.bk.arrIdmLow[-1]
                    self.idmLBar = self.bk.arrIdmLBar[-1]
                self.bk.updateLastH(self.H, i)

            if lo <= self.L:
                self.L, self.LBar = lo, i
                if self.bk.arrIdmHigh:
                    self.idmHigh  = self.bk.arrIdmHigh[-1]
                    self.idmHBar  = self.bk.arrIdmHBar[-1]
                self.bk.updateLastL(self.L, i)

            # 4) بناء OBS/OBD من [b-2] (+ Mother[b-2] لو isb[2]) — مع إصلاح off-by-one
            self._build_obs_obd_from_b2(i, df)

            # 5) خريطة الهيكل + EXT + IDM take
            bos_up,bos_dn,choch_up,choch_dn = self._structure_mapping(hi,lo,cl,i)
            self._bos_up[i]=bos_up; self._bos_dn[i]=bos_dn
            self._choch_up[i]=choch_up; self._choch_dn[i]=choch_dn

            # 6) أول ملامسة IDM
            touch_s, touch_d = self._first_touch_idm(i, df)
            self._idm_touch_supply[i] = touch_s
            self._idm_touch_demand[i] = touch_d

            # 7) تنظيف المناطق المكنوسة بالكامل
            for z in list(self.zones):
                z.right_index = i
                swept = (z.kind == "SUPPLY" and hi >= z.top + self.cfg.eps) or \
                        (z.kind == "DEMAND" and lo <= z.bottom - self.cfg.eps)
                if swept:
                    self.zones.remove(z)

        # مخرجات رقمية فقط
        signals_df = pd.DataFrame({
            "bos_up": self._bos_up,
            "bos_dn": self._bos_dn,
            "choch_up": self._choch_up,
            "choch_dn": self._choch_dn,
            "idm_touch_supply": self._idm_touch_supply,
            "idm_touch_demand": self._idm_touch_demand,
        }, index=df.index)

        levels_df = pd.DataFrame({
            "lastH": [self.lastH]*n,
            "lastL": [self.lastL]*n,
            "H":     [self.H]*n,
            "L":     [self.L]*n,
            "idmHigh": [self.idmHigh]*n,
            "idmLow":  [self.idmLow]*n,
            "motherHigh": [self.motherHigh]*n,
            "motherLow":  [self.motherLow]*n,
        }, index=df.index)

        zones_df = pd.DataFrame([z.__dict__ for z in self.zones]) if self.zones else \
                   pd.DataFrame(columns=["left_index","right_index","top","bottom","kind","label","state"])

        alerts_df = pd.DataFrame(self.alerts) if self.alerts else \
                    pd.DataFrame(columns=["bar","type","price","box"])

        bookkeeping_df = pd.DataFrame({
            "arrLastH":   [self.bk.arrLastH],
            "arrLastL":   [self.bk.arrLastL],
            "arrIdmHigh": [self.bk.arrIdmHigh],
            "arrIdmLow":  [self.bk.arrIdmLow],
            "arrBCLabel": [self.bk.arrBCLabel],
            "arrHLLabel": [self.bk.arrHLLabel],
        })

        return {
            "signals": signals_df,
            "levels": levels_df,
            "zones": zones_df,
            "alerts": alerts_df,
            "bookkeeping": bookkeeping_df
        }


# =========================
# Binance Fetcher
# =========================
class BinanceFetcher:
    def __init__(self):
        if Client is None:
            raise RuntimeError("python-binance غير مثبت. ثبّت الحزمة: pip install python-binance")
        api_key    = _get_secret("BINANCE_API_KEY")
        api_secret = _get_secret("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "[BINANCE] مفاتيح غير موجودة: ضعها في API_KEYS أعلى الملف "
                "أو كمُتغيّرات بيئة BINANCE_API_KEY/BINANCE_API_SECRET."
            )
        self.client = Client(api_key=api_key, api_secret=api_secret)

    def get_top_usdt_perps(self, max_symbols: int = 30) -> List[str]:
        try:
            tickers = self.client.futures_ticker()
            usdt = [t for t in tickers if str(t.get("symbol", "")).endswith("USDT")]
            top = sorted(usdt, key=lambda t: float(t.get("quoteVolume", 0.0)), reverse=True)
            return [t["symbol"] for t in top[:max_symbols]]
        except Exception as e:
            print(f"[BINANCE] symbols error: {e}")
            return []

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            kl = self.client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
            if not kl:
                return None
            df = pd.DataFrame(kl, columns=[
                'timestamp','open','high','low','close','volume','close_time',
                'qv','trades','tb_base','tb_quote','ignore'
            ])
            df = df[['timestamp','open','high','low','close','volume']].copy()
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"[BINANCE] fetch error {symbol}: {e}")
            return None


# =========================
# ماسح Binance Futures
# =========================
class FuturesScanner:
    def __init__(self, cfg: Settings, timeframe="1m", limit=500, max_symbols=200,
                 symbol_override: str = "", verbose: bool = False, recent_bars: int = 1,
                 drop_last: bool = False):
        self.cfg = cfg
        self.timeframe = timeframe
        self.limit = limit
        self.max_symbols = max_symbols
        self.symbol_override = symbol_override.strip().upper()
        self.verbose = verbose
        self.recent_bars = max(1, int(recent_bars))
        self.drop_last = bool(drop_last)
        self.fetcher = BinanceFetcher() if Client else None
        self.tg = TelegramNotifier() if cfg.tg_enable else None

    def _fmt_price(self, p: float) -> str:
        if p >= 100: return f"{p:.2f}"
        if p >= 1:   return f"{p:.4f}"
        return f"{p:.6f}"

    def _print_summary(self, sym: str, tf: str, out: Dict[str, pd.DataFrame]):
        sig = out["signals"]
        lookback = min(300, len(sig))
        sub = sig.tail(lookback)
        bos_up_cnt = int(sub["bos_up"].sum())
        bos_dn_cnt = int(sub["bos_dn"].sum())
        cho_up_cnt = int(sub["choch_up"].sum())
        cho_dn_cnt = int(sub["choch_dn"].sum())
        idm_s_cnt  = int(sub["idm_touch_supply"].sum())
        idm_d_cnt  = int(sub["idm_touch_demand"].sum())
        zones_cnt  = len(out["zones"])
        last_levels = out["levels"].tail(1).iloc[0]
        print(f"[{sym} {tf}] لا تنبيهات. ملخص {lookback} شمعات — "
              f"BOS↑:{bos_up_cnt} BOS↓:{bos_dn_cnt} | "
              f"CHoCh↑:{cho_up_cnt} CHoCh↓:{cho_dn_cnt} | "
              f"IDM Touch S:{idm_s_cnt} D:{idm_d_cnt} | "
              f"Zones:{zones_cnt} | "
              f"lastH:{self._fmt_price(float(last_levels['lastH']))} "
              f"lastL:{self._fmt_price(float(last_levels['lastL']))}")

    def run(self):
        if not self.fetcher:
            print("[SCAN] python-binance غير متاح. ثبّت الحزمة: pip install python-binance")
            return

        # تحديد الرموز
        if self.symbol_override:
            symbols = [self.symbol_override]
        else:
            symbols = self.fetcher.get_top_usdt_perps(self.max_symbols)

        if not symbols:
            print("[SCAN] لم يتم الحصول على أي رموز. تحقّق من مفاتيح API أو من اتصال الإنترنت.")
            return

        tf = self.timeframe
        lim = self.limit

        for sym in symbols:
            df = self.fetcher.fetch_ohlcv(sym, tf, lim)
            if df is None or df.empty:
                if self.verbose:
                    print(f"[{sym} {tf}] لا توجد بيانات OHLCV (df فارغ).")
                continue

            # خيار تجاهل الشمعة الأخيرة (مضاهاة Pine على الشموع المؤكدة فقط)
            if self.drop_last and len(df) > 0:
                df = df.iloc[:-1].copy()

            engine = SMCIndicator(self.cfg)
            out = engine.run(df)
            alerts_df = out["alerts"]

            last_bar_idx = len(df) - 1
            if not alerts_df.empty:
                recent_cut = last_bar_idx - (self.recent_bars - 1)
                recent_alerts = alerts_df[alerts_df["bar"] >= recent_cut]
            else:
                recent_alerts = pd.DataFrame()

            if not recent_alerts.empty:
                for _, a in recent_alerts.tail(20).iterrows():
                    typ = str(a.get("type", "ALERT"))
                    price = float(a.get("price", 0.0))
                    msg = f"[{sym} {tf}] {typ} @ {self._fmt_price(price)}"
                    print(msg)
                    if self.tg:
                        self.tg.send(f"{self.cfg.tg_title_prefix}: {msg}")
            else:
                if self.verbose:
                    self._print_summary(sym, tf, out)


# =========================
# MAIN (argparse)
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="SMC Binance Scanner",
        description="SMC (Pine 1:1) + Binance Futures Scanner + Telegram"
    )
    parser.add_argument("--timeframe", "-t", default="1m",
                        help="1m, 5m, 15m, 1h, 4h (افتراضي: 15m)")
    parser.add_argument("--limit", "-l", type=int, default=1000,
                        help="عدد الشموع لكل رمز (افتراضي: 1000)")
    parser.add_argument("--max-symbols", "-n", type=int, default=300,
                        help="عدد أعلى رموز USDT بالحجم (افتراضي: 30)")
    parser.add_argument("--mitigation", choices=["WICK","CLOSE"], default="WICK",
                        help="طريقة الـMitigation: WICK للذيل أو CLOSE للإغلاق")
    parser.add_argument("--tg", action="store_true", default=False,
                        help="إرسال التنبيهات لتلغرام (يتطلب TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID)")
    parser.add_argument("--symbol", "-s", default="",
                        help="رمز واحد للتشخيص (مثال: BTCUSDT). اتركه فارغًا لاختيار أعلى الأزواج بالحجم")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="اطبع ملخصًا حتى لو لا توجد تنبيهات")
    parser.add_argument("--recent", type=int, default=300,
                        help="اطبع تنبيهات آخر N شموع (افتراضي 1 = آخر شمعة فقط)")
    parser.add_argument("--drop-last", action="store_true", default=False,
                        help="تجاهل آخر شمعة (لتطابق أشد مع Pine على الشموع المؤكدة)")
    args = parser.parse_args()

    # فحص وجود الحزمة python-binance
    if Client is None:
        print("[!] تحتاج لتثبيت python-binance: pip install python-binance")
        sys.exit(1)

    # تذكير فقط: سنسحب المفاتيح من API_KEYS أو من البيئة تلقائيًا
    if not _get_secret("BINANCE_API_KEY") or not _get_secret("BINANCE_API_SECRET"):
        print("[!] ملاحظة: لم يتم العثور على مفاتيح Binance في API_KEYS أو البيئة. "
              "ضعها في أعلى الملف (API_KEYS) أو كمتغيرات بيئة قبل التشغيل.")
        sys.exit(1)

    cfg = Settings(
        eps=1e-10,
        enable_alert_bos=True,
        enable_alert_choch=True,
        enable_alert_idm_touch=True,
        poi_type_mother_bar=True,
        mitigation_mode=args.mitigation,
        merge_ratio=0.10,
        tg_enable=bool(args.tg),
        tg_title_prefix="SMC Alert"
    )

    scanner = FuturesScanner(cfg,
                             timeframe=args.timeframe,
                             limit=args.limit,
                             max_symbols=args.max_symbols,
                             symbol_override=args.symbol,
                             verbose=args.verbose,
                             recent_bars=args.recent,
                             drop_last=args.drop_last)
    scanner.run()