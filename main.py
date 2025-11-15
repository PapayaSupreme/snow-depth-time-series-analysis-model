from pandas import read_csv
from pathlib import Path
from models.naive_seasonal import naive_seasonal
from models.sarima import sarima
from utils.cleaner import clean_all
from utils.accuracy import accuracy

def main():
    print("cleaning raw samples, please wait...")
    clean_all()
    print("Cleaned samples.\n\n")

    folder = Path("./cleaned v2")
    choices = []
    i = 1
    print("Enter data source to use model on: ")
    for file in folder.iterdir():
        if file.is_file():
            choices.append(file.name)
            print(i, "-", choices[-1])
            i+=1
    print("0 - ALL OF THEM :)")

    choice = int(input())
    while 0 > choice or choice >= len(choices)+1:
        print("out of bounds. try again pls")
        choice = int(input("Enter file name to use model on: "))
    if choice == 0:
        filenames = choices
    else:
        filenames = [choices[choice-1]]

    dfs = {}
    for filename in filenames:
        dfs[filename] = read_csv("./cleaned v2/" + filename)

    for k in dfs:
        avg_HS_after_gapfill = dfs[k]["HS_after_gapfill"].abs().mean()

        df_naive = naive_seasonal(dfs[k])
        print("Naive method Accuracy for station", k, ": ", accuracy(df_naive, "HS_naive"),
              "after normalizing (mae / avg snow depth): ", accuracy(df_naive, "HS_naive")
              / avg_HS_after_gapfill)
        print()

        df_sarima = sarima(dfs[k], True)
        print("SARIMA model Accuracy for station", k, ": ", accuracy(df_sarima, "HS_sarima"),
              "after normalizing (mae / avg snow depth): ", accuracy(df_sarima, "HS_sarima")
              / avg_HS_after_gapfill)
        print()

if __name__ == "__main__":
    main()