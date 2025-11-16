from pandas import read_csv
from pathlib import Path

from models.naive_seasonal import naive_seasonal
from models.sarima import sarima
from models.arima import *
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
    exit = False
    while not exit:
        print("Choose a model to train")
        print("1. NAIVE SEASONAL")
        print("2. (AR)(I)(MA)")
        print("3. SARIMA(X)")
        print("0. EXIT")
        choice = -1
        while 0>choice or choice>3 :
            choice = int(input())
        match choice:
            case 0:
                exit = True

            case 1:
                for k in dfs:
                    avg_HS_after_gapfill = dfs[k]["HS_after_gapfill"].abs().mean()

                    df_naive = naive_seasonal(dfs[k])
                    print(k, "- naive Accuracy: ", accuracy(df_naive, "HS_naive"),
                          "- Normalized: ", accuracy(df_naive, "HS_naive")
                          / avg_HS_after_gapfill)
                    print()

            case 2:
                correct = False
                p, d, q = 0, 0, 0
                while not correct:
                    p = int(input("Choose p (AR) parameter"))
                    d = int(input("Choose d (I) parameter"))
                    q = int(input("Choose q (MA) parameter"))
                    print(f"Chose {p} {d} {q} as parameter, correct ?")
                    print("0. NO")
                    print("1. YES")
                    correct = int(input()) == 1
                for k in dfs:
                    results_111, mae_111 = rolling_seasonal_cv_arima(
                        dfs[k],
                        p, d, q,
                        min_train_seasons=10  # or 5 if you want earlier validation starts
                    )

                    print(f"=== {k} ARIMA({p}, {d}, {q}) Rolling Validation ===")
                    print(results_111.head())
                    print("Global MAE:", mae_111)
            case 3:
                heaviness = -1
                print("Choose heaviness of grid: ")
                print("0. 212")
                print("1. 90")
                print("2. 30")
                print("3. 7")
                print("4. NO GRID")
                while 0 > heaviness or heaviness > 4:
                    heaviness = int(input())
                simple_diff = -1
                print("Simplified Differentials ?: ")
                print("0. NO")
                print("1. YES")
                while simple_diff != 0 and simple_diff != 1:
                    simple_diff = int(input())
                for k in dfs:
                    avg_HS_after_gapfill = dfs[k]["HS_after_gapfill"].abs().mean()
                    df_sarima = sarima(dfs[k], heaviness, simple_diff == 1)
                    print(k, "- SARIMA Accuracy: ", accuracy(df_sarima, "HS_sarima"),
                          "- Normalized: ", accuracy(df_sarima, "HS_sarima")
                          / avg_HS_after_gapfill)
                    print()


if __name__ == "__main__":
    main()