import os
import re

def best_arima_hyperparameters(stations):
    """
    Iterate through every file in the arima results
    and finds the best hyperparameters set for each station

    :param stations: (list) list of station data as filenames (str)

    :return: (dict) result with structure:
                            {
                                "col_de_porte_daily.txt":
                                {
                                    "p": 2,
                                    "d": 1,
                                    "q": 2,
                                    "mae": 0.92,
                                    "file": "2 1 2.txt"
                                    }
                                }

    """
    folder = "./computed/arima/"
    best = {}
    stations_set = set(stations)

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue

        # filename is "p d q.txt"
        try:
            p, d, q = map(int, filename[:-4].split())
        except ValueError:
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        for station in stations_set:
            if f"=== {station} ARIMA" not in content:
                continue

            match = re.search(
                rf"=== {re.escape(station)}.*?Global MAE:\s*([0-9.]+)",
                content,
                flags=re.DOTALL
            )
            if not match:
                continue

            mae = float(match.group(1))

            if station not in best or mae < best[station]["mae"]:
                best[station] = {
                    "p": p,
                    "d": d,
                    "q": q,
                    "mae": mae,
                    "file": filename,
                }

    return best
