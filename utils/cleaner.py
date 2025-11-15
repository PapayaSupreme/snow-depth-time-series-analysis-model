from pandas import read_csv, to_datetime
from pathlib import Path

def clean_file(filename: str) -> None:
    """
        Creates into the "./cleaned v2" the file stripped of
        dates not between 01-11 to 31-05 included.
    :param filename: name of the file, with extension
    :return: none
    """
    df = read_csv("./cleaned v1/" + filename)
    date_col = df.columns[2]
    df[date_col] = to_datetime(df[date_col])

    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day

    mask = ((df["month"] >= 11) | (df["month"] <= 5)) & ~((df["month"] == 2) & (df["day"] == 29))

    filtered = df[mask]
    filtered = filtered.drop(columns=["month", "day"])

    filtered.to_csv("./cleaned v2/" + filename, index=False)

def clean_all() -> None:
    """
        Cleans every file present in folder "./cleaned v1/"
        by calling clean_file().
    """
    folder = Path("./cleaned v1")
    for file in folder.iterdir():
        if file.is_file():
            clean_file(str(file)[11:])
