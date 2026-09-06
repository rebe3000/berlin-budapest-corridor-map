#!/usr/bin/env python3
"""
Berlin–Budapest corridor overview map (OSM).

- Motorways + branches to Vienna, Warsaw, Ostrava
- Trunk/primary connectors filling motorway gaps (amber)
- highway=services within ~200 m of a motorway
- Mikulášov marked as planned comfort break
- Plot in EPSG:3857 with equal aspect (undistorted)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CITIES = {
    "Berlin": (13.3694, 52.5256),
    "Dresden": (13.7320, 51.0400),
    "Prague": (14.4350, 50.0830),
    "Brno": (16.6120, 49.1910),
    "Bratislava": (17.1070, 48.1580),
    "Budapest": (19.0830, 47.5000),
    "Vienna": (16.3738, 48.2082),
    # User-requested corridor cities
    "Görlitz": (14.9885, 51.1552),
    "Zittau": (14.8076, 50.8977),
    "Liberec": (15.0562, 50.7671),
    "Pasohlávky": (16.5436, 48.9030),
    "Mikulov": (16.6378, 48.8056),
    # Waypoints so the corridor follows real motorway arcs (not straight chords)
    "Győr": (17.6651, 47.6875),  # Bratislava–Budapest via M1/D1
    "Břeclav": (16.8820, 48.7590),  # Brno–Vienna via D2 / A5
    "Poysdorf": (16.6300, 48.6700),  # A5 corridor AT–CZ
}

# Cities drawn as hub labels (waypoints help geometry but stay unlabeled)
LABEL_CITIES = (
    "Berlin",
    "Dresden",
    "Görlitz",
    "Zittau",
    "Liberec",
    "Prague",
    "Brno",
    "Pasohlávky",
    "Mikulov",
    "Bratislava",
    "Budapest",
    "Vienna",
)

# Main product + southern / Lusatian branches (no Warsaw / Ostrava)
HUB_SEGMENTS = [
    ("Berlin", "Dresden"),
    ("Dresden", "Prague"),
    ("Dresden", "Görlitz"),
    ("Görlitz", "Zittau"),
    ("Zittau", "Liberec"),
    ("Liberec", "Prague"),
    ("Prague", "Brno"),
    ("Brno", "Bratislava"),
    ("Brno", "Pasohlávky"),
    ("Pasohlávky", "Mikulov"),
    ("Mikulov", "Poysdorf"),
    ("Brno", "Břeclav"),
    ("Břeclav", "Poysdorf"),
    ("Poysdorf", "Vienna"),
    ("Bratislava", "Vienna"),
    ("Bratislava", "Győr"),
    ("Győr", "Budapest"),
]

BUFFER_KM = 50.0
SERVICE_MAX_DIST_M = 200.0  # “directly on the highway”
CONNECTOR_TAGS = ("trunk", "trunk_link", "primary", "primary_link")
# Planned comfort break (Odpočívka Mikulášov, D1 ~km 95.7; midpoint of both carriageways)
MIKULASOV = {
    "name": "Mikulášov",
    "lon": 15.3393,
    "lat": 49.5374,
    "label": "Mikulášov\n(planned comfort break)",
}
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {
    "User-Agent": "FridayForFuture-FlixCaseStudy/1.0 (corridor overview map; educational)",
    "Accept": "application/json",
}


def _seg_key(a: str, b: str) -> str:
    return f"{a}_{b}"


def _hub_lines() -> list[LineString]:
    return [LineString([CITIES[a], CITIES[b]]) for a, b in HUB_SEGMENTS]


def _corridor_polygon():
    lines = gpd.GeoSeries(_hub_lines(), crs="EPSG:4326").to_crs(epsg=3857)
    poly = unary_union([g.buffer(BUFFER_KM * 1000.0) for g in lines.geometry])
    return gpd.GeoSeries([poly], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]


def _segment_bboxes():
    """Yield (key, south, west, north, east) for each hub segment."""
    for a, b in HUB_SEGMENTS:
        seg = LineString([CITIES[a], CITIES[b]])
        g = gpd.GeoSeries([seg], crs="EPSG:4326").to_crs(epsg=3857)
        poly = g.buffer(BUFFER_KM * 1000.0).iloc[0]
        poly4326 = gpd.GeoSeries([poly], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        minx, miny, maxx, maxy = poly4326.bounds
        yield _seg_key(a, b), miny, minx, maxy, maxx


def _overpass_query(query: str, *, label: str) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, 6):
        for url in OVERPASS_URLS:
            try:
                host = url.split("/")[2]
                print(f"    {label}: try {attempt}/5 → {host}")
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers=HEADERS,
                    timeout=(30, 300),
                )
                if resp.status_code == 429:
                    wait = 30 * attempt
                    print(f"    rate-limited; sleep {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    print(f"    HTTP {resp.status_code}: {resp.text[:180].replace(chr(10), ' ')}")
                    time.sleep(12)
                    continue
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = min(45, 8 * attempt)
                print(f"    {label}: {type(exc).__name__}: {exc}; sleep {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Overpass failed for {label}: {last_err}")


def _ways_to_gdf(data: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        rows.append(
            {
                "osm_id": el["id"],
                "highway": tags.get("highway"),
                "name": tags.get("name"),
                "geometry": LineString(coords),
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "highway", "name", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _pois_to_gdf(data: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if tags.get("highway") != "services":
            continue
        if el["type"] == "node" and "lat" in el:
            geom = Point(el["lon"], el["lat"])
        elif "center" in el:
            geom = Point(el["center"]["lon"], el["center"]["lat"])
        elif el["type"] == "way" and "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) < 3:
                continue
            geom = Polygon(coords).centroid
        else:
            continue
        rows.append(
            {
                "osm_id": el["id"],
                "highway": "services",
                "name": tags.get("name"),
                "geometry": geom,
            }
        )
    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "highway", "name", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _fetch_ways_by_segment(
    *,
    cache_subdir: str,
    combined_name: str,
    overpass_highway_values: list[str],
    poly,
    label_prefix: str,
) -> gpd.GeoDataFrame:
    combined_path = CACHE_DIR / combined_name
    if combined_path.exists():
        print(f"Loading cached {label_prefix} from {combined_path}")
        gdf = gpd.read_file(combined_path)
        if "highway" in gdf.columns:
            gdf = gdf[gdf["highway"].isin(overpass_highway_values)].copy()
        return gdf

    seg_dir = CACHE_DIR / cache_subdir
    seg_dir.mkdir(exist_ok=True)
    tag_union = "|".join(overpass_highway_values)

    print(f"Downloading {label_prefix} from Overpass ({len(HUB_SEGMENTS)} segments)...")
    parts = []
    for key, south, west, north, east in _segment_bboxes():
        seg_path = seg_dir / f"{key}.gpkg"
        # Migrate old numbered caches for the first five main segments
        if not seg_path.exists() and cache_subdir == "motorway_segments":
            legacy_map = {
                "Berlin_Dresden": "seg_1.gpkg",
                "Dresden_Prague": "seg_2.gpkg",
                "Prague_Brno": "seg_3.gpkg",
                "Brno_Bratislava": "seg_4.gpkg",
                "Bratislava_Budapest": "seg_5.gpkg",
            }
            legacy_name = legacy_map.get(key)
            if legacy_name:
                legacy = seg_dir / legacy_name
                if legacy.exists():
                    legacy.rename(seg_path)

        if seg_path.exists():
            print(f"  {label_prefix} {key} (cached)")
            parts.append(gpd.read_file(seg_path))
            continue

        print(f"  {label_prefix} {key}...")
        query = f"""
[out:json][timeout:180];
(
  way["highway"~"^({tag_union})$"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
);
out geom;
"""
        data = _overpass_query(query, label=f"{label_prefix}-{key}")
        gdf = _ways_to_gdf(data)
        gdf = gdf[gdf["highway"].isin(overpass_highway_values)].copy()
        gdf.to_file(seg_path, driver="GPKG")
        print(f"    → {len(gdf)} ways")
        parts.append(gdf)
        time.sleep(8)

    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    corridor = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    out = gpd.clip(out, corridor).drop_duplicates(subset=["geometry"])
    out.to_file(combined_path, driver="GPKG")
    print(f"Saved {len(out)} {label_prefix} features → {combined_path}")
    return out


def _filter_connectors_in_motorway_gaps(
    motorways: gpd.GeoDataFrame,
    connectors: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Keep trunk (+ limited primary) that bridge gaps along hub links — fast path."""
    if connectors.empty:
        return connectors

    path = CACHE_DIR / "connectors_gap_fill_v9.gpkg"
    if path.exists():
        print(f"Loading cached gap-fill connectors from {path}")
        return gpd.read_file(path)

    print("Filtering connectors to motorway gaps...")
    mw = motorways.to_crs(epsg=3857)
    cn = connectors.to_crs(epsg=3857)
    hubs = gpd.GeoSeries(_hub_lines(), crs="EPSG:4326").to_crs(epsg=3857)
    hub_buf = unary_union([g.buffer(30_000) for g in hubs.geometry])
    mw_union = unary_union(list(mw.geometry))
    mw_buf = mw_union.buffer(60)

    # --- trunk: along full hub corridor ---
    trunk = cn[cn["highway"].isin(("trunk", "trunk_link"))].copy()
    trunk = trunk[trunk.intersects(hub_buf)]
    print(f"  trunk candidates: {len(trunk)}")

    # --- primary: only near motorway-absent hub stretches (avoids city street clutter + speed) ---
    gap_buffers = []
    for line in _hub_lines():
        line_m = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
        n = max(10, int(line_m.length / 12_000))
        for i in range(n + 1):
            pt = line_m.interpolate(i / n, normalized=True)
            if pt.distance(mw_union) > 1_200:
                gap_buffers.append(pt.buffer(10_000))
    primary = cn[cn["highway"].isin(("primary", "primary_link"))].copy()
    if gap_buffers:
        gap_zone = unary_union(gap_buffers)
        primary = primary[primary.intersects(gap_zone)]
    else:
        primary = primary.iloc[0:0]
    print(f"  primary in gap zones: {len(primary)}")

    cand = gpd.GeoDataFrame(pd.concat([trunk, primary], ignore_index=True), crs="EPSG:3857")
    if cand.empty:
        empty = connectors.iloc[0:0].copy()
        empty.to_file(path, driver="GPKG")
        return empty

    mw_mask = gpd.GeoDataFrame(geometry=[mw_buf], crs="EPSG:3857")
    off = gpd.overlay(cand, mw_mask, how="difference", keep_geom_type=False)
    if off.empty:
        empty = connectors.iloc[0:0].copy()
        empty.to_file(path, driver="GPKG")
        return empty

    off = off.explode(index_parts=False).reset_index(drop=True)
    off = off[~off.geometry.is_empty & off.geometry.notna()].copy()
    off = off[off.geometry.length >= 120].copy()
    out = off.to_crs(epsg=4326)
    out.to_file(path, driver="GPKG")
    print(f"Gap-fill connectors kept: {len(out)} → {path}")
    return out


def _merge_connectors(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if a.empty and b.empty:
        return a
    if a.empty:
        return b
    if b.empty:
        return a
    out = gpd.GeoDataFrame(pd.concat([a, b], ignore_index=True), crs="EPSG:4326")
    return out.drop_duplicates(subset=["geometry"])


def _load_or_fetch_services(poly, motorways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    poi_path = CACHE_DIR / "services_on_motorway.gpkg"
    if poi_path.exists():
        print(f"Loading cached on-motorway services from {poi_path}")
        return gpd.read_file(poi_path)

    seg_dir = CACHE_DIR / "service_on_mw_segments"
    seg_dir.mkdir(exist_ok=True)

    print("Downloading highway=services from Overpass...")
    parts = []
    for key, south, west, north, east in _segment_bboxes():
        seg_path = seg_dir / f"{key}.gpkg"
        if seg_path.exists():
            print(f"  Services {key} (cached)")
            parts.append(gpd.read_file(seg_path))
            continue

        print(f"  Services {key}...")
        query = f"""
[out:json][timeout:120];
(
  node["highway"="services"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
  way["highway"="services"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
  relation["highway"="services"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});
);
out center;
"""
        data = _overpass_query(query, label=f"services-{key}")
        gdf = _pois_to_gdf(data)
        gdf.to_file(seg_path, driver="GPKG")
        print(f"    → {len(gdf)} sites")
        parts.append(gdf)
        time.sleep(8)

    if not parts:
        empty = gpd.GeoDataFrame(columns=["osm_id", "highway", "name", "geometry"], crs="EPSG:4326")
        empty.to_file(poi_path, driver="GPKG")
        return empty

    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    corridor = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    out = gpd.clip(out, corridor).drop_duplicates(subset=["geometry"])

    # Keep only sites within SERVICE_MAX_DIST_M of a motorway centreline
    print(f"Filtering services to ≤{SERVICE_MAX_DIST_M:.0f} m of motorway...")
    mw_m = motorways.to_crs(epsg=3857)
    mw_union = unary_union(mw_m.geometry.values)
    svc_m = out.to_crs(epsg=3857)
    dist = svc_m.geometry.distance(mw_union)
    out = out.loc[dist.values <= SERVICE_MAX_DIST_M].copy()
    out.to_file(poi_path, driver="GPKG")
    meta = {
        "count": int(len(out)),
        "tags": ["services"],
        "max_dist_m": SERVICE_MAX_DIST_M,
        "buffer_km": BUFFER_KM,
    }
    (CACHE_DIR / "services_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(out)} on-motorway services → {poi_path}")
    return out


def _plot(
    motorways: gpd.GeoDataFrame,
    connectors: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
) -> None:
    mw_m = motorways.to_crs(epsg=3857)
    cn_m = connectors.to_crs(epsg=3857) if not connectors.empty else connectors
    services_m = services.to_crs(epsg=3857) if not services.empty else services
    cities_gdf = gpd.GeoDataFrame(
        {"name": list(LABEL_CITIES)},
        geometry=[Point(CITIES[n]) for n in LABEL_CITIES],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    mik_m = gpd.GeoSeries(
        [Point(MIKULASOV["lon"], MIKULASOV["lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    bounds_parts = [mw_m]
    if not connectors.empty:
        bounds_parts.append(cn_m)
    xmin, ymin, xmax, ymax = gpd.GeoDataFrame(pd.concat(bounds_parts, ignore_index=True)).total_bounds
    pad = 40_000
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad
    width_m = xmax - xmin
    height_m = ymax - ymin
    # Equal aspect: figure size matches geographic width/height (Web Mercator metres)
    fig_w = 12.0
    fig_h = max(7.0, fig_w * height_m / width_m)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=150)
    ax.set_facecolor("#eef2f5")
    fig.patch.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    try:
        import contextily as cx

        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.PositronNoLabels,
            zoom="auto",
            attribution=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Basemap skipped ({exc}); using plain background")

    if not connectors.empty:
        cn_m.plot(ax=ax, color="#b7791f", linewidth=1.05, alpha=0.9, zorder=2)

    if not mw_m.empty:
        mw_m.plot(ax=ax, color="#2c5282", linewidth=1.2, alpha=0.88, zorder=3)

    if not services_m.empty:
        services_m.plot(
            ax=ax,
            color="#c05621",
            markersize=14,
            marker="o",
            edgecolor="white",
            linewidth=0.3,
            alpha=0.85,
            zorder=4,
        )

    cities_gdf.plot(
        ax=ax,
        color="#1a202c",
        markersize=55,
        marker="s",
        zorder=5,
        edgecolor="white",
        linewidth=0.8,
    )
    offsets = {
        "Berlin": (12_000, 16_000),
        "Dresden": (-48_000, 10_000),
        "Görlitz": (14_000, 14_000),
        "Zittau": (-48_000, -6_000),
        "Liberec": (14_000, -16_000),
        "Prague": (-62_000, 10_000),
        "Brno": (14_000, 14_000),
        "Pasohlávky": (-62_000, 8_000),
        "Mikulov": (14_000, -18_000),
        "Bratislava": (12_000, 12_000),
        "Vienna": (-52_000, -10_000),
        "Budapest": (12_000, -18_000),
    }
    for _, row in cities_gdf.iterrows():
        x, y = row.geometry.x, row.geometry.y
        dx, dy = offsets.get(row["name"], (12_000, 10_000))
        ax.annotate(
            row["name"],
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=9,
            fontweight="bold",
            color="#1a202c",
            zorder=6,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )

    # Planned comfort break — Mikulášov
    mik_m.plot(
        ax=ax,
        color="#c53030",
        markersize=120,
        marker="*",
        zorder=7,
        edgecolor="white",
        linewidth=0.6,
    )
    mx, my = mik_m.iloc[0].x, mik_m.iloc[0].y
    ax.annotate(
        MIKULASOV["label"],
        xy=(mx, my),
        xytext=(mx + 25_000, my + 18_000),
        fontsize=8.5,
        fontweight="bold",
        color="#c53030",
        zorder=8,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color="#c53030", lw=0.8),
    )

    ax.set_title(
        "Berlin–Budapest corridor (+ Vienna)\n"
        "Motorways, gap connectors, on-motorway services — Mikulášov comfort break",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
        color="#1a202c",
    )
    ax.set_axis_off()

    legend_handles = [
        Line2D([0], [0], color="#2c5282", lw=2, label="Motorway / motorway link"),
        Line2D(
            [0],
            [0],
            color="#b7791f",
            lw=2,
            label="Trunk / primary (gaps & approaches)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#c05621",
            markersize=8,
            label=f"Service station on motorway (≤{SERVICE_MAX_DIST_M:.0f} m)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#c53030",
            markersize=14,
            label="Mikulášov — planned comfort break",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#1a202c",
            markersize=8,
            label="City",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        edgecolor="#cbd5e0",
        fontsize=8,
    )

    n_svc = 0 if services.empty else len(services_m)
    ax.text(
        0.99,
        0.01,
        f"OpenStreetMap · EPSG:3857 equal aspect · highway=services ≤{SERVICE_MAX_DIST_M:.0f} m · "
        f"{n_svc} services · Mikulášov D1 km ~95.7",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#4a5568",
    )

    fig.tight_layout()
    png = OUT_DIR / "berlin_budapest_highways_services.png"
    pdf = OUT_DIR / "berlin_budapest_highways_services.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main() -> None:
    poly = _corridor_polygon()

    motorways = _fetch_ways_by_segment(
        cache_subdir="motorway_segments",
        combined_name="motorways_edges_v9.gpkg",
        overpass_highway_values=["motorway", "motorway_link"],
        poly=poly,
        label_prefix="motorways",
    )
    connectors_raw = _fetch_ways_by_segment(
        cache_subdir="connector_segments",
        combined_name="connectors_edges_v9.gpkg",
        overpass_highway_values=list(CONNECTOR_TAGS),
        poly=poly,
        label_prefix="connectors",
    )
    connectors = _filter_connectors_in_motorway_gaps(motorways, connectors_raw)
    services = _load_or_fetch_services(poly, motorways)
    _plot(motorways, connectors, services)


if __name__ == "__main__":
    main()
