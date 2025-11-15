from pandas import read_csv
from pathlib import Path
from models.naive_seasonal import naive_seasonal, naive_accuracy
from cleaner import clean_all

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
        df_with_naive = naive_seasonal(dfs[k])
        print("Naive method Accuracy for station", k, ": ", naive_accuracy(df_with_naive),
              "after normalizing (mae / avg snow depth): ", naive_accuracy(df_with_naive) / df_with_naive["HS_after_gapfill"].abs().mean())

main()