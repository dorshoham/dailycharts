"""
Daily Financial Charts Email
============================

Generates two charts from live Yahoo Finance data and emails them to you,
with the charts shown directly inside the email body.

  Chart 1 - Rolling SPX/VIX Beta (20-day window)
  Chart 2 - 261-Day Rolling Beta of SPX vs 10-Year Yield Changes

Designed to run automatically once a day via GitHub Actions.

How it decides whether to send (self-healing)
----------------------------------------------
GitHub's free scheduler is not punctual - it often runs jobs late, and can
skip a run entirely. So instead of demanding an exact time, the workflow
triggers several times each morning and this script sends the email on the
FIRST run that meets both of these conditions:

  * it is 10:00 or later in Israel, and
  * no email has been sent yet today.

The "already sent today" memory is a small file (last_sent.txt) kept in the
repository. This means: if the 10am run is late or skipped, a later run
simply catches up - and you still get exactly one email per day.

A manual run always sends immediately and does not touch that memory file.

Environment variables (set as GitHub Secrets)
---------------------------------------------
  GMAIL_ADDRESS       - the Gmail address that sends the email
  GMAIL_APP_PASSWORD  - a 16-character Gmail "app password" (NOT your login password)
  RECIPIENT           - where to send it (optional; defaults to dscrist7@gmail.com)
  FORCE_SEND          - "1" to skip all checks (set automatically on manual runs)
"""

import os
import sys
import ssl
import smtplib
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no screen needed - we only save image files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:  # very old Python fallback
    ZoneInfo = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _close_series(ticker, **kwargs):
    """Download one ticker and return its Close price as a clean Series."""
    data = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):       # newer yfinance returns a frame
        close = close.iloc[:, 0]
    return close.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 - Rolling SPX/VIX Beta
# ─────────────────────────────────────────────────────────────────────────────
def chart_rolling_vix_beta(out_path="rolling_spx_vix_beta.png"):
    ROLLING_WINDOW = 20
    START_DATE = "2024-01-01"
    END_DATE = datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(["^GSPC", "^VIX"], start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    spx = raw["Close"]["^GSPC"].dropna()
    vix = raw["Close"]["^VIX"].dropna()
    df = pd.DataFrame({"SPX": spx, "VIX": vix}).dropna()

    spx_ret = df["SPX"].pct_change() * 100
    vix_chg = df["VIX"].diff()
    roll_cov = spx_ret.rolling(ROLLING_WINDOW).cov(vix_chg)
    roll_var = spx_ret.rolling(ROLLING_WINDOW).var()
    df["RollingBeta"] = roll_cov / roll_var
    df = df.dropna()

    fig, ax1 = plt.subplots(figsize=(19.2, 9.67))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    color_beta = "#E63946"
    ax1.plot(df.index, df["RollingBeta"], color=color_beta,
             linewidth=1.4, label="Rolling Beta")
    ax1.set_ylabel("Beta (VIX points per 1% SPX move)",
                   color="#222222", fontsize=11, labelpad=10)
    ax1.tick_params(axis="y", colors="#222222")
    ax1.tick_params(axis="x", colors="#222222")
    ax1.spines[["top", "right", "left", "bottom"]].set_color("#CCCCCC")
    ax1.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="-")

    beta_min = df["RollingBeta"].min()
    beta_max = df["RollingBeta"].max()
    ax1.set_ylim(beta_min - 0.1, beta_max + 0.3)

    ax2 = ax1.twinx()
    ax2.set_facecolor("white")
    color_spx = "#1D6FA4"
    ax2.plot(df.index, df["SPX"], color=color_spx,
             linewidth=1.4, label="SPX", alpha=0.9)
    ax2.set_ylabel("SPX Price", color="#222222", fontsize=11, labelpad=10)
    ax2.tick_params(axis="y", colors="#222222")
    ax2.spines[["top", "right", "left", "bottom"]].set_color("#CCCCCC")

    ax1.grid(True, color="#E5E5E5", linewidth=0.5, linestyle="-")
    ax2.grid(False)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center", color="#222222")
    ax1.set_xlabel("Date", color="#222222", fontsize=11, labelpad=8)
    ax1.set_title("Rolling SPX/VIX Beta", color="#222222", fontsize=14, pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left", facecolor="white",
               edgecolor="#CCCCCC", labelcolor="#222222", fontsize=10, framealpha=0.9)
    ax2.legend(lines2, labels2, loc="upper right", facecolor="white",
               edgecolor="#CCCCCC", labelcolor="#222222", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 - 261-Day Rolling Beta of SPX vs 10-Year Yield Changes
# ─────────────────────────────────────────────────────────────────────────────
def chart_rolling_10y_beta(out_path="rolling_beta_261d_spx_10y.png"):
    START_DATE = "1960-01-01"
    WINDOW = 261

    spx = _close_series("^GSPC", start=START_DATE)
    tnx = _close_series("^TNX", start=START_DATE)

    df = pd.DataFrame({"SPX": spx, "Yield10Y": tnx}).sort_index().dropna()
    df["SPX_ret"] = df["SPX"].pct_change() * 100
    df["Yield_chg"] = df["Yield10Y"].diff()
    df = df.dropna()

    roll_cov = df["Yield_chg"].rolling(WINDOW).cov(df["SPX_ret"])
    roll_var = df["Yield_chg"].rolling(WINDOW).var()
    df["Beta"] = roll_cov / roll_var
    df_plot = df.dropna(subset=["Beta"])

    fig, ax = plt.subplots(figsize=(19, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(df_plot.index, df_plot["Beta"], color="#CC44CC", linewidth=1.0, zorder=3)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="-", zorder=2)
    ax.grid(True, color="#DDDDDD", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")

    ax.tick_params(colors="#222222", labelsize=9)
    ax.set_ylabel("Beta", color="#222222", fontsize=10)
    ax.set_xlabel("Date", color="#222222", fontsize=10, labelpad=6)
    ax.set_title("261-Day Rolling Beta of SPX vs 10-Year Yield Changes",
                 color="#222222", fontsize=12, pad=10)

    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", color="#222222")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────
CHARTS = [
    ("Rolling SPX/VIX Beta", chart_rolling_vix_beta, "chart1"),
    ("261-Day Rolling Beta: SPX vs 10-Year Yield", chart_rolling_10y_beta, "chart2"),
]


def build_and_send_email(results, sender, password, recipient):
    """results: list of (title, cid, image_path_or_None, error_or_None)."""
    today = datetime.now().strftime("%A, %d %B %Y")
    msg = MIMEMultipart("related")
    msg["Subject"] = f"Daily Financial Charts - {today}"
    msg["From"] = sender
    msg["To"] = recipient

    body = [
        '<div style="font-family:Arial,Helvetica,sans-serif;color:#222;max-width:1000px;">',
        f'<h2 style="color:#222;">Daily Financial Charts</h2>',
        f'<p style="color:#666;">{today}</p>',
    ]
    for title, cid, img_path, error in results:
        body.append(f'<h3 style="color:#222;margin-top:28px;">{title}</h3>')
        if img_path:
            body.append(
                f'<img src="cid:{cid}" alt="{title}" '
                f'style="width:100%;max-width:1000px;height:auto;border:1px solid #eee;">'
            )
        else:
            body.append(
                f'<p style="color:#B00020;">Could not generate this chart today.<br>'
                f'<span style="color:#888;font-size:12px;">{error}</span></p>'
            )
    body.append(
        '<p style="color:#999;font-size:12px;margin-top:32px;">'
        'Sent automatically. Data from Yahoo Finance.</p></div>'
    )
    msg.attach(MIMEText("\n".join(body), "html"))

    for title, cid, img_path, error in results:
        if img_path:
            with open(img_path, "rb") as f:
                img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(img_path))
            msg.attach(img)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    print(f"Email sent to {recipient}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
SEND_HOUR_IL = 10                 # send at 10:00 Israel time or later
STATE_FILE = "last_sent.txt"      # remembers the date of the last email sent


def _israel_now():
    tz = ZoneInfo("Asia/Jerusalem") if ZoneInfo is not None else None
    return datetime.now(tz)


def _already_sent_today(today_str):
    try:
        with open(STATE_FILE) as f:
            return f.read().strip() == today_str
    except FileNotFoundError:
        return False


def main():
    sender = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("RECIPIENT", "dscrist7@gmail.com")
    force_send = os.environ.get("FORCE_SEND") == "1"

    now_il = _israel_now()
    today_str = now_il.strftime("%Y-%m-%d")

    # Decide whether to send now (a manual run skips all of these checks).
    if not force_send:
        if _already_sent_today(today_str):
            print(f"Email already sent today ({today_str}) - nothing to do.")
            return
        if now_il.hour < SEND_HOUR_IL:
            print(f"Too early in Israel (currently {now_il:%H:%M}) - will try again later.")
            return
        print(f"It is {now_il:%H:%M} in Israel and no email sent yet today - sending now.")

    # Generate each chart; one failing must not stop the others.
    results = []
    for title, fn, cid in CHARTS:
        try:
            print(f"Generating: {title}")
            path = fn()
            results.append((title, cid, path, None))
            print(f"  ok -> {path}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append((title, cid, None, repr(exc)))

    if not sender or not password:
        print("\nNo Gmail credentials set - DRY RUN, no email sent.")
        print("Charts that were generated:",
              [r[2] for r in results if r[2]])
        return

    build_and_send_email(results, sender, password, recipient)

    # Record today's date so later runs today don't send a second email.
    # (A manual/forced run does not write this, so it can't block the real run.)
    if not force_send:
        with open(STATE_FILE, "w") as f:
            f.write(today_str)
        print(f"Recorded send date: {today_str}")


if __name__ == "__main__":
    main()
