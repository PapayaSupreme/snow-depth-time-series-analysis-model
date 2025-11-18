from pandas import read_csv
from pathlib import Path

from models.naive_seasonal import rolling_naive_seasonal
from models.sarima import rolling_seasonal_sarima
from models.arima import rolling_seasonal_arima
from utils.best_hyperparameters import best_arima_hyperparameters
from utils.cleaner import clean_all
from os import makedirs

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
        choice = int(input())
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
        print("3. (AR)(I)(MA) - all hyperparameters")
        print("4. (AR)(I)(MA) - choose best hyperparameters")
        print("5. (S)ARIMA ~X")
        print("0. EXIT")
        choice = -1
        while 0>choice or choice>5 :
            print("out of bounds. try again pls")
            choice = int(input())
        match choice:
            case 0:
                exit = True

            case 1:
                for k in dfs:
                    results, mae = rolling_naive_seasonal(
                        dfs[k],
                        min_train_seasons=10
                    )

                    print(f"=== {k} NAIVE SEASONAL Rolling Validation ===")
                    print(results.tail(5))
                    print("Global MAE:", mae, "\n")

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
                    results, mae, season, predicted, pct_error = rolling_seasonal_arima(
                        dfs[k],
                        p, d, q,
                        min_train_seasons=10
                    )

                    print(f"=== {k} ARIMA({p}, {d}, {q}) Rolling Validation ===")
                    print(results.tail(5))
                    pct_error = ((predicted - season) / season) * 100.0
                    print("Global: MAE:", mae.round(3), "mean:", season.round(3), "predicted:", predicted.round(3), "%:", pct_error.round(3))
            case 3:
                makedirs("./computed/arima/", exist_ok=True)
                p_list = [0, 1, 2, 3]
                d_list = [0, 1, 2]
                q_list = [0, 1, 2, 3]

                for p in p_list:
                    for d in d_list:
                        for q in q_list:
                            lines = []

                            for k in dfs:
                                results, mae = rolling_seasonal_arima(
                                    dfs[k],
                                    p, d, q,
                                    min_train_seasons=10
                                )

                                lines.append(f"=== {k} ARIMA({p}, {d}, {q}) Rolling Validation ===\n")
                                lines.append(results.tail(5).to_string() + "\n")
                                lines.append(f"Global MAE: {mae}\n\n")

                            filename = f"./computed/arima/{p} {d} {q}.txt"
                            with open(filename, "w", encoding="utf-8") as f:
                                f.writelines(lines)

                            print(f"Saved report to {filename}")
            case 4:
                best_models = best_arima_hyperparameters(filenames)
                """averages = {}
                for filename in filenames:
                    averages[filename] = [float(dfs[filename]["HS_after_gapfill"].mean()),
                                          float(dfs[filename]["HS_after_gapfill"].mean()
                                          * best_models[filename]["mae"]),
                                          float((dfs[filename]["HS_after_gapfill"].mean()
                                          - dfs[filename]["HS_after_gapfill"].mean()
                                          * best_models[filename]["mae"]))
                                          ]"""

                for station, info in best_models.items():
                    print(station, info)
                    #print(averages[station])
            case 5:
                print("NOTE: ARIMA PARAMETERS ARE (p = 1, d = 1, q = 1)")
                correct = False
                P, D, Q = 0, 0, 0
                while not correct:
                    P = int(input("Choose Q (AR) parameter"))
                    D = int(input("Choose D (I) parameter"))
                    Q = int(input("Choose Q (MA) parameter"))
                    print(f"Chose {P} {D} {Q} as parameter, correct ?")
                    print("0. NO")
                    print("1. YES")
                    correct = int(input()) == 1
                S = -1
                print("Choose seasonal heaviness of grid: ")
                while 0 > S or S > 212:
                    S = int(input())
                simple_diff = -1
                print("Simplified Differentials ?: ")
                print("0. NO")
                print("1. YES")
                while simple_diff != 0 and simple_diff != 1:
                    simple_diff = int(input())
                for k in dfs:
                    results, mae = rolling_seasonal_sarima(
                        dfs[k],
                        1, 1, 1,
                        P, D, Q, S,
                        simple_diff,
                        min_train_seasons=10
                    )

                    print(f"=== {k} SARIMA(1, 1, 1)({P}, {D}, {Q}, {S}) Rolling Validation ===")
                    print(results.tail(5))
                    print("Global MAE:", mae, "\n")


if __name__ == "__main__":
    main()