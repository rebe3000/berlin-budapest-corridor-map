# Berlin–Budapest corridor map

Python + Jupyter notebook that builds an OpenStreetMap overview of motorways, trunk/primary gap connectors, on-motorway service stations, and the planned **Mikulášov** comfort break.

## Files

| File | Role |
|------|------|
| `Berlin_Budapest_Corridor_Map.ipynb` | Jupyter entrypoint |
| `plot_berlin_budapest_corridor.py` | Map generator (imported by the notebook) |
| `requirements.txt` | Python dependencies |
| `berlin_budapest_highways_services.png` / `.pdf` | Last exported map |

## Run locally

```bash
python3 -m pip install -r requirements.txt
jupyter notebook Berlin_Budapest_Corridor_Map.ipynb
```

Then run all cells. First Overpass download can take several minutes; results cache under `cache/`.

## Data

© OpenStreetMap contributors ([ODbL](https://www.openstreetmap.org/copyright)).
