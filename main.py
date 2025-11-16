from pandas import read_csv
from pathlib import Path

from models.naive_seasonal import rolling_naive_seasonal
from models.sarima import *
from models.arima import *
from utils.cleaner import clean_all

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
                    results_111, mae_111 = rolling_naive_seasonal(
                        dfs[k],
                        min_train_seasons=30
                    )

                    print(f"=== {k} NAIVE SEASONAL Rolling Validation ===")
                    print(results_111.head())
                    print("Global MAE:", mae_111)

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
                        min_train_seasons=30
                    )

                    print(f"=== {k} ARIMA({p}, {d}, {q}) Rolling Validation ===")
                    print(results_111.head())
                    print("Global MAE:", mae_111)
            case 3:
                heaviness = -1
                print("Choose seasonal heaviness of grid: ")
                while 0 > heaviness or heaviness > 212:
                    heaviness = int(input())
                simple_diff = -1
                print("Simplified Differentials ?: ")
                print("0. NO")
                print("1. YES")
                while simple_diff != 0 and simple_diff != 1:
                    simple_diff = int(input())
                for k in dfs:
                    results_111, mae_111 = rolling_seasonal_cv_sarima(
                        dfs[k],
                        1, 1, 1,
                        1, 0, 1, heaviness,
                        simple_diff,
                        min_train_seasons=30
                    )

                    print(f"=== {k} SARIMA(1, 1, 1)(1, 0, 1, {heaviness}) Rolling Validation ===")
                    print(results_111)
                    print("Global MAE:", mae_111)


if __name__ == "__main__":
    main()