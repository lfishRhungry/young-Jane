# wash_and_get_features.py reads scanning results of goscanner(modified)
# and generate features
import pandas as pd
import argparse
import os

# -------------------------------parse args
parser = argparse.ArgumentParser(description="wash goscanner results and get features")
parser.add_argument(
    "--dir", "-d", help="directory of goscanner outputs files", type=str, default="./"
)
parser.add_argument(
    "--output",
    "-o",
    help="filename of output features(csv)",
    type=str,
    default="features.csv",
)
args = parser.parse_args()

# ------------------------------------wash data

data_hosts = pd.read_csv(os.path.join(args.dir, "hosts.csv"))
data_verbose = pd.read_csv(os.path.join(args.dir, "tls_verbose.csv"))

merge_data = pd.merge(data_hosts, data_verbose, how="inner", on="id")

# filt out fail connection and connections with ALPN
merge_data = merge_data[
    merge_data["resultString"].str.contains("SUCCESS")
    & (~merge_data["server_hello_extensions"].str.contains("16,", na=False))
]

# merge_data.to_excel("hosts_verbose.xlsx", index=False)

# --------------------------------extract features we need
col_data = merge_data.loc[
    :,
    [
        "port",
        "protocol",
        "cipher",
        "cert_valid",
        "server_hello_length",
        "server_hello_extensions_length",
        "server_hello_extensions",
    ],
]

# convert cipher suites from hex to dec
col_data["cipher"] = col_data["cipher"].apply(int, base=16)

# parse server_hello_extensions to new columns
col_data["status_request_5"] = (
    col_data["server_hello_extensions"]
    .str.contains("5,", na=False)
    .map(lambda x: 1 if x else 0)
)
col_data["ec_point_formats_11"] = (
    col_data["server_hello_extensions"]
    .str.contains("11,", na=False)
    .map(lambda x: 1 if x else 0)
)
col_data["session_ticket_35"] = (
    col_data["server_hello_extensions"]
    .str.contains("35,", na=False)
    .map(lambda x: 1 if x else 0)
)
col_data["supported_versions_43"] = (
    col_data["server_hello_extensions"]
    .str.contains("43,", na=False)
    .map(lambda x: 1 if x else 0)
)
col_data["psk_key_exchange_modes_45"] = (
    col_data["server_hello_extensions"]
    .str.contains("45,", na=False)
    .map(lambda x: 1 if x else 0)
)
col_data["key_share_51"] = (
    col_data["server_hello_extensions"]
    .str.contains("51,", na=False)
    .map(lambda x: 1 if x else 0)
)
col_data["renegotiation_info_65281"] = (
    col_data["server_hello_extensions"]
    .str.contains("65281,", na=False)
    .map(lambda x: 1 if x else 0)
)

feature_data = col_data.drop(columns="server_hello_extensions")
# feature_data.to_excel("features.xlsx", index=False)
feature_data.to_csv(args.output, index=False, float_format="%.0f")
