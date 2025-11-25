from pandas import read_csv
from pathlib import Path

from models.gru import rolling_seasonal_gru
from models.naive_seasonal import rolling_naive_seasonal
from models.sarima import rolling_seasonal_sarima
from models.arima import rolling_seasonal_arima
from models.prophet_model import rolling_seasonal_prophet
from utils.best_hyperparameters import best_arima_hyperparameters
from utils.cleaner import clean_all
from os import makedirs
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
import pandas as pd


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
        print("3. (AR)(I)(MA) - try all hyperparameters combination")
        print("4. (AR)(I)(MA) - find best hyperparameters (from files)")
        print("5. (AR)(I)(MA) - find best hyperparameters (from acf / pacf)")
        print("6. (S)ARIMA ~X")
        print("7. PROPHET")
        print("8. display data per year")
        print("9. display data per day")
        print("10. GRU")
        print("0. EXIT")
        choice = -1
        while 0>choice or choice>10 :
            choice = int(input())
        match choice:
            case 0:
                exit = True

            case 1:
                for k in dfs:
                    is_whole = "whole" in k
                    results, mae, nmae, season, predicted, pct_error = rolling_naive_seasonal(
                        dfs[k],
                        is_whole,
                    )#TODO: bigru lstm rnns

                    print(f"=== {k} NAIVE Rolling Validation ===")
                    print(results.tail(5))
                    pct_error = ((predicted - season) / season) * 100.0
                    print("Global: MAE:", mae.round(3), "NMAE:", nmae.round(3), "mean:", season.round(3), "predicted:", predicted.round(3),
                          "%:", pct_error.round(3))
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
                    """results, mae, nmae, season, predicted, pct_error = arima_train_test_split(
                        dfs[k],
                        p, d, q,
                    )

                    print(f"=== {k} ARIMA({p}, {d}, {q}) train test split Validation ===")
                    print(results.tail(5))
                    pct_error = ((predicted - season) / season) * 100.0
                    print("Global: MAE:", mae.round(3), "NMAE:", nmae.round(3), "mean:", season.round(3), "predicted:", predicted.round(3), "%:", pct_error.round(3))
                    print()"""
                    is_whole =  "whole" in k

                    results, mae, nmae, season, predicted, pct_error = rolling_seasonal_arima(
                        dfs[k],
                        p, d, q,
                        is_whole,
                    )

                    print(f"=== {k} ARIMA({p}, {d}, {q}) Rolling Validation ===")
                    print(results.tail(5))
                    pct_error = ((predicted - season) / season) * 100.0
                    print("Global: MAE:", mae.round(3), "NMAE:", nmae.round(3), "mean:", season.round(3), "predicted:",
                          predicted.round(3), "%:", pct_error.round(3))
                    print()
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
                                results, mae, nmae, season, predicted, pct_error = rolling_seasonal_arima(
                                    dfs[k],
                                    p, d, q,
                                    min_train_seasons=10
                                )

                                lines.append(f"=== {k} ARIMA({p}, {d}, {q}) Rolling Validation ===\n")
                                lines.append(results.tail(5).to_string() + "\n")
                                lines.append(f"Global - MAE: {mae.round(3)}  NMAE: {nmae.round(3)}  mean: {season.round(3)}  predicted: {predicted.round(3)} %: {pct_error.round(3)}\n\n")

                            filename = f"./computed/arima/{p} {d} {q}.txt"
                            with open(filename, "w", encoding="utf-8") as f:
                                f.writelines(lines)

                            print(f"Saved report to {filename}")
            case 4:
                best_models = best_arima_hyperparameters(filenames)

                for station, info in best_models.items():
                    print(station, info)
            case 5:
                for k in dfs:
                    series = dfs[k]["HS_after_gapfill"].dropna()

                    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

                    plot_acf(series, ax=ax[0], lags=50)
                    plot_pacf(series, ax=ax[1], lags=50)

                    ax[0].set_title("ACF - HS_after_gapfill")
                    ax[1].set_title("PACF - HS_after_gapfill")

                    plt.tight_layout()
                    plt.title(k)
                    plt.show()
                    result = adfuller(series)

                    print("ADF Statistic of", k,":", result[0])
                    print("p-value:", result[1])
                    print("Lags used:", result[2])
                    print("Number of observations:", result[3])

                    for key, value in result[4].items():
                        print("Critical Value (%s): %.3f" % (key, value))

            case 6:
                print("NOTE: ARIMA PARAMETERS ARE (p = 1, d = 0, q = 1)")
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
                    results, mae, nmae, season, predicted, pct_error = rolling_seasonal_sarima(
                        dfs[k],
                        1, 1, 1,
                        P, D, Q, S,
                        simple_diff,
                        min_train_seasons=10
                    )

                    print(f"=== {k} SARIMA(1, 0, 1)({P}, {D}, {Q}, {S}) Rolling Validation ===")
                    print(results.tail(5))
                    pct_error = ((predicted - season) / season) * 100.0
                    print("Global: MAE:", mae.round(3), "NMAE:", nmae.round(3), "mean:", season.round(3), "predicted:",
                          predicted.round(3), "%:", pct_error.round(3))
                    print()
            case 7:
                for k in dfs:
                    results, mae, nmae, season, predicted, pct_error = rolling_seasonal_prophet(
                        dfs[k],
                        min_train_seasons=10,
                    )

                    print(f"=== {k} PROPHET Rolling Validation ===")
                    print(results.tail(5))
                    pct_error = ((predicted - season) / season) * 100.0
                    print("Global: MAE:", mae.round(3), "NMAE:", nmae.round(3), "mean:", season.round(3), "predicted:",
                          predicted.round(3), "%:", pct_error.round(3))
                    print()
            case 8:
                for k in dfs:
                    df_copy = dfs[k].copy()
                    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                    df_copy = df_copy.set_index('Date').sort_index()

                    m = df_copy.index.month
                    season_mask = (m >= 11) | (m <= 5)
                    df_winter = df_copy[season_mask]

                    months = df_winter.index.month
                    season_year = pd.Series(df_winter.index.year, index=df_winter.index)
                    season_year[months <= 5] -= 1

                    seasonal_means = df_winter.groupby(season_year)['HS_after_gapfill'].mean()

                    plt.figure(figsize=(12, 6))
                    plt.plot(seasonal_means.index, seasonal_means.values, marker='o', linestyle='-', linewidth=2, markersize=8)
                    plt.xlabel('Season Year')
                    plt.ylabel('Mean Snow Depth (cm)')
                    plt.title(f'HS per season of {k}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            case 9:
                for k in dfs:
                    df_copy = dfs[k].copy()
                    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
                    df_copy = df_copy.set_index('Date').sort_index()

                    plt.figure(figsize=(14, 6))
                    plt.plot(df_copy.index, df_copy['HS_after_gapfill'], linewidth=1, alpha=0.8)
                    plt.xlabel('Date')
                    plt.ylabel('Snow Depth (cm)')
                    plt.title(f'HS per day of {k}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            case 10:
                print("1. PROD")
                print("0. TEST")
                choice = int(input("test or prod"))
                if choice == 0:
                    for k in dfs:
                        results = rolling_seasonal_gru(
                            dfs[k], is_whole=False,
                            seq_len=60,
                            hidden_dim=32,
                            epochs=50,
                            patience=5,
                            min_train_seasons=10)
                        print(k)
                        print(results)
                else:
                    for k in dfs:
                        results = rolling_seasonal_gru(
                            dfs[k], is_whole=False,
                            seq_len=120,
                            hidden_dim=128,
                            epochs=200,
                            patience=15,
                            min_train_seasons=10)
                        print(results)


if __name__ == "__main__":
    main()