# agent_fraud.py
"""
Module: agent_fraud.py
Deskripsi: Logika deteksi fraud berbasis rule + statistik.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi dan validasi kolom penting (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "transactionid" in cols: rename_map[cols["transactionid"]] = "TransactionID"
    if "date" in cols: rename_map[cols["date"]] = "Date"
    if "time" in cols: rename_map[cols["time"]] = "Time"
    if "vendor" in cols: rename_map[cols["vendor"]] = "Vendor"
    if "amount" in cols: rename_map[cols["amount"]] = "Amount"
    if "account" in cols: rename_map[cols["account"]] = "Account"
    if "employeeid" in cols: rename_map[cols["employeeid"]] = "EmployeeID"
    if "invoicenumber" in cols: rename_map[cols["invoicenumber"]] = "InvoiceNumber"
    if "description" in cols: rename_map[cols["description"]] = "Description"

    df = df.rename(columns=rename_map)
    # minimal check
    if not (("Date" in df.columns) and ("Amount" in df.columns) and (("TransactionID" in df.columns) or ("InvoiceNumber" in df.columns))):
        raise ValueError("File tidak valid. Pastikan ada kolom Date, Amount, dan TransactionID atau InvoiceNumber.")
    # parse types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    # ensure cols exist
    for c in ["TransactionID","Time","Vendor","Account","EmployeeID","InvoiceNumber","Description"]:
        if c not in df.columns:
            df[c] = ""
    return df.fillna("")

def detect_duplicate_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """Temukan InvoiceNumber yang muncul lebih dari sekali."""
    if "InvoiceNumber" not in df.columns:
        return pd.DataFrame()
    dup = df[df["InvoiceNumber"]!=""].groupby("InvoiceNumber").filter(lambda x: len(x)>1)
    if dup.empty:
        return pd.DataFrame()
    summary = dup.groupby("InvoiceNumber").agg(Count=("TransactionID","count"), TotalAmount=("Amount","sum")).reset_index()
    return summary.sort_values("TotalAmount", ascending=False)

def detect_amount_spikes(df: pd.DataFrame, z_thresh=3.0) -> pd.DataFrame:
    """Deteksi outlier per Account menggunakan z-score."""
    outliers = []
    for acc, g in df.groupby("Account"):
        amounts = g["Amount"].astype(float)
        if len(amounts) < 5:
            continue
        mu = amounts.mean()
        sigma = amounts.std(ddof=0)
        if sigma == 0:
            continue
        z = (amounts - mu) / sigma
        for idx, zi in zip(g.index, z):
            if abs(zi) >= z_thresh:
                row = g.loc[idx].to_dict()
                row["Zscore"] = float(zi)
                row["AccountMean"] = float(mu)
                row["AccountStd"] = float(sigma)
                outliers.append(row)
    return pd.DataFrame(outliers)

def detect_weekend_night_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Deteksi transaksi weekend atau di luar jam kerja (06-20 default)."""
    res = []
    for _, r in df.iterrows():
        date = r.get("Date")
        added = False
        if pd.notna(date):
            if date.weekday() >= 5:
                out = r.to_dict(); out["Reason"]="Weekend"; res.append(out); added=True
        tstr = str(r.get("Time",""))
        try:
            if tstr and ":" in tstr:
                hh = int(tstr.split(":")[0])
                if hh < 6 or hh > 20:
                    if not added:
                        out = r.to_dict(); out["Reason"]="Diluar jam kerja"; res.append(out)
        except Exception:
            pass
    if res:
        return pd.DataFrame(res).drop_duplicates(subset=["TransactionID"])
    return pd.DataFrame()

def detect_employee_anomalies(df: pd.DataFrame, min_tx=5) -> pd.DataFrame:
    """Karyawan dengan frekuensi atau nilai total transaksi tidak wajar."""
    summary = df.groupby("EmployeeID").agg(Count=("TransactionID","count"), TotalAmount=("Amount","sum")).reset_index()
    mean_total = summary["TotalAmount"].mean() if not summary.empty else 0
    suspects = summary[(summary["Count"]>=min_tx) | (summary["TotalAmount"]>=mean_total*2)]
    return suspects.sort_values(by=["Count","TotalAmount"], ascending=False)

def detect_many_small_invoices_same_vendor(df: pd.DataFrame, same_day_count=4, small_amount=200.0) -> pd.DataFrame:
    """Vendor yang pada hari sama mengeluarkan banyak invoice kecil (indikator splitting)."""
    if df["Date"].isnull().all():
        return pd.DataFrame()
    df2 = df.copy(); df2["DateOnly"] = pd.to_datetime(df2["Date"]).dt.date
    grp = df2.groupby(["Vendor","DateOnly"]).filter(lambda x: (x["Amount"]<=small_amount).sum()>=same_day_count)
    if grp.empty:
        return pd.DataFrame()
    summary = grp.groupby(["Vendor","DateOnly"]).agg(Count=("TransactionID","count"), TotalAmount=("Amount","sum")).reset_index()
    return summary.sort_values("Count", ascending=False)

def detect_threshold_amounts(df: pd.DataFrame, thresholds=[9999,99999,1000000]) -> pd.DataFrame:
    """Transaksi dengan amount tepat pada nilai ambang (biasanya modus manipulasi)."""
    hits = df[df["Amount"].isin(thresholds)]
    return hits

def score_alerts(dup_df, spikes_df, weekend_df, emp_df, many_small_df, thresh_df):
    """Gabungkan temuan menjadi tabel alerts dengan skor prioritas sederhana."""
    alerts = []
    if not dup_df.empty:
        for _, r in dup_df.iterrows():
            alerts.append({"Type":"DuplicateInvoice","Key": r["InvoiceNumber"], "Message": f"Invoice {r['InvoiceNumber']} muncul {int(r['Count'])} kali, total {r['TotalAmount']:.2f}", "Score": 80 + min(20, int(r['TotalAmount']/1000))})
    if not spikes_df.empty:
        for _, r in spikes_df.iterrows():
            alerts.append({"Type":"AmountSpike","Key": r.get("TransactionID",""), "Message": f"Transaksi {r.get('TransactionID','')} bernilai {r.get('Amount'):.2f} (z={r.get('Zscore'):.2f}) di account {r.get('Account')}", "Score": 70 + min(30, abs(r.get("Zscore",0))*5)})
    if not weekend_df.empty:
        for _, r in weekend_df.iterrows():
            alerts.append({"Type":"TimingAnomaly","Key": r.get("TransactionID",""), "Message": f"Transaksi {r.get('TransactionID','')} pada {r.get('Date')} waktu {r.get('Time')} ({r.get('Reason')})", "Score": 50})
    if not emp_df.empty:
        for _, r in emp_df.iterrows():
            alerts.append({"Type":"EmployeePattern","Key": r.get("EmployeeID",""), "Message": f"Karyawan {r.get('EmployeeID')} melakukan {int(r.get('Count'))} transaksi, total {r.get('TotalAmount'):.2f}", "Score": 60 + min(40, int(r.get('Count')) )})
    if not many_small_df.empty:
        for _, r in many_small_df.iterrows():
            alerts.append({"Type":"ManySmallInvoices","Key": f"{r.get('Vendor')}|{r.get('DateOnly')}", "Message": f"Vendor {r.get('Vendor')} pada {r.get('DateOnly')} mengeluarkan {int(r.get('Count'))} invoice kecil, total {r.get('TotalAmount'):.2f}", "Score": 65})
    if not thresh_df.empty:
        for _, r in thresh_df.iterrows():
            alerts.append({"Type":"ThresholdAmount","Key": r.get("TransactionID",""), "Message": f"Transaksi {r.get('TransactionID','')} bernilai ambang {r.get('Amount'):.2f}", "Score": 55})
    alerts = sorted(alerts, key=lambda x: x["Score"], reverse=True)
    return pd.DataFrame(alerts)
