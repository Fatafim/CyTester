from typing import Callable
import multiprocessing as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tabulate
from joblib import Parallel, delayed
from compute import compute
import seaborn as sns
from itertools import chain
sns.set_theme()

def compute_parallel_wrapper(instance, data):
    return instance.compute_parallel(data)
def compute_parallel_wrapper2(args):
    instance, data = args
    return instance.compute_parallel(data)

def merge_dicts(list_of_dicts):
    return dict(chain(*[d.items() for d in list_of_dicts]))
class CyTester:
    def __init__(self,
                 data: pd.DataFrame,
                 decision: Callable,
                 indicators: pd.DataFrame,
                 starting_index: int = 1,
                 leverage: float = 1,
                 one_lot_worth: float = 1,
                 lot: float = 1,
                 cap: float = 100,
                 static_sl: float = 100_000_000_000,
                 static_tp: float = 100_000_000_000,
                 perc_sl: float = 100_000_000_000,
                 perc_tp: float = 100_000_000_000,
                 const_spread: float = 0,
                 **kwargs):

        if data.columns.nlevels > 1:
            for column in data.columns:
                assert not data[column].lt(0).any(), "Close cannot contain negative values"
                self.many = True
            print("DUPA")
        else:
            self.many = False

            assert not data.lt(0).any()[0], "Close cannot contain negative values"

        self.orders = None
        self.data = data
        self.cap = cap
        self.decision = decision
        self.indicators = indicators
        self.starting_index = starting_index
        self.out = data
        self.leverage = leverage
        self.lot = lot
        self.multiplier = one_lot_worth * lot * leverage
        self.static_sl = static_sl
        self.static_tp = static_tp
        self.perc_sl = perc_sl
        self.perc_tp = perc_tp
        self.const_spread = const_spread / 2
        self.many_outputs = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def compute(self, data: pd.DataFrame, indicators):
        return compute(data.values.reshape(1, -1)[0],
                       self.decision,
                       indicators=indicators.values,
                       cap=self.cap,
                       multiplier=self.multiplier,
                       start_index=self.starting_index,
                       static_sl=self.static_sl,
                       static_tp=self.static_tp,
                       perc_sl=self.perc_sl,
                       perc_tp=self.perc_tp,
                       const_spread=self.const_spread
                       )

    def compute_parallel(self, data: pd.DataFrame):
        result = {}
        for instrument, close in data.columns:
            orders = self.compute(data[instrument], self.indicators[instrument])
            result[instrument] = self.combine_results(self.out[instrument], orders)
        return result


    def run(self):
        if not self.many:
            self.orders = self.compute(self.data, self.indicators)
            self.out = self.combine_results(self.out, self.orders)
        else:

            results = Parallel(n_jobs=mp.cpu_count())(
                delayed(compute_parallel_wrapper2)((self, self.data[instrument].to_frame()))
                for instrument in self.data.columns
            )
            self.many_outputs = merge_dicts(results)

    def stats_single(self, instrument):
        if not self.many:
            out = self.out
        else:
            out = self.many_outputs[instrument]
        out = out.assign(pct_returns=out.capital.pct_change() + 1)

        lasting = (out
                   .groupby("id")
                   .count()
                   .iloc[:, 0]
                   .drop(index=[-1]))

        trades = (
            out[["returns", "id"]]
            .groupby("id")
            .sum()
            .drop(index=[-1]))

        lasting_winning = lasting[trades.where(trades > 0).dropna().index]
        lasting_losing = lasting[trades.where(trades <= 0).dropna().index]

        average_lasting_winning = lasting_winning.mean()
        average_lasting_losing = lasting_losing.mean()

        average_trade = trades.mean()[0]
        median_trade = trades.median()[0]
        q5_trade = trades.quantile(.05)[0]
        q95_trade = trades.quantile(.95)[0]

        winning_trades = trades.where(trades > 0)
        losing_trades = trades.where(trades <= 0)

        average_winning = winning_trades.mean()[0]
        average_losing = losing_trades.mean()[0]

        how_many_winning = winning_trades.count()[0]
        how_many_losing = losing_trades.count()[0]

        pct_return = (out["capital"].values[-1] - out["capital"].values[0]) / out["capital"].values[0]
        total_return = (out["capital"].values[-1] - out["capital"].values[0])

        przerwa = "-------------------"

        _headers = ['Total return:', '% return:', przerwa,
                   "Start capital:", "End capital:", przerwa,
                   "Average trade:", "0.05 quantile:", "Median trade:", "0.95 quantile:", przerwa,
                   "Average winning trade:", "Average losing trade:", "Average time winning", "Average time losing", przerwa,
                   "Winning trades:", "Losing trades:", "Win rate:"]

        _statistics = [round(total_return, 2), round(pct_return, 4), przerwa,
                      round(self.cap, 2), round(out["capital"].iloc[-1], 2), przerwa,
                      round(average_trade, 2), round(q5_trade, 2), round(median_trade, 2), round(q95_trade, 2), przerwa,
                      round(average_winning, 2), round(average_losing, 2), average_lasting_winning, average_lasting_losing, przerwa,
                      how_many_winning, how_many_losing, round(how_many_winning / (how_many_losing + how_many_winning), 2)]

        self.statistics = dict(zip(_headers, _statistics))
        self.statistics.pop(przerwa)
        result = tabulate.tabulate({
            "Category": _headers,
            "Value": _statistics
        }, headers='keys')
        print(result)
        return self.statistics

    def stats(self, instrument: list = []):
        if not self.many:
            self.stats_single(instrument)
        else:
            returns = {}
            for key in self.many_outputs:
                returns[key] = (self.many_outputs[key]["capital"].values[-1] - self.many_outputs[key]["capital"].values[
                    0]) / self.many_outputs[key]["capital"].values[0]

            result = pd.DataFrame(returns, index=["% returns"])

    def bar(self):
        assert self.many, "You must pass more than one instrument to use barplots"

        for key in self.many_outputs:
            plt.bar(key, (self.many_outputs[key]["capital"].values[-1] - self.many_outputs[key]["capital"].values[0]) /
                    self.many_outputs[key]["capital"].values[0])

        plt.show()

    def plot(self, indicators_to_plot: list = [], instrument: str = ""):

        if self.many:
            assert len(instrument) > 0, "Type in the instrument you want to display"
            out = self.many_outputs[instrument]
        else:
            out = self.out
            print("______________________________________")

        hold = CyTester(out["close"].to_frame(),
                          decision=lambda close, indicators, i, previous_state: 1,
                          indicators=self.indicators,  # to nie ma żadnego znaczenia
                          cap=self.cap,
                          starting_index=self.starting_index,
                          leverage=self.leverage,
                          multiplier=self.multiplier,
                          lot=self.lot,
                          const_spread=self.const_spread)
        hold.run()

        first_trade = np.where(out.id == 0)[0] - 1
        first_trade = first_trade[0]

        cumprod_close = np.cumprod(hold.out["capital"].pct_change() + 1)
        cumprod_capital = np.cumprod(out["capital"].pct_change() + 1).fillna(0)
        cumprod_capital[0] = None

        fig, ax = plt.subplots(3, 1, figsize=(20, 15))
        ax0 = ax[0]
        ax1 = ax[1]
        ax2 = ax[2]
        ax0.title.set_text("% return vs hold")

        ax0.plot(cumprod_close, label='Hold', c='r')
        ax0.plot(cumprod_capital, label='Tested strategy', c='g')
        ax0.axhline(1, color='grey', linestyle='--', alpha=0.5)

        greater_eq = cumprod_close >= cumprod_capital
        less = cumprod_close < cumprod_capital
        ax0.fill_between(out.index, cumprod_capital, 1,
                         where=(greater_eq & (cumprod_capital < 1) & (cumprod_close < 1)), color='red', alpha=0.2,
                         edgecolor='red')
        ax0.fill_between(out.index, cumprod_close, 1, where=(less & (cumprod_close < 1) & (cumprod_capital < 1)),
                         color='red', alpha=0.2, edgecolor='red')

        sum_returns = (
            out.loc[:, ["id", "returns"]]
            .groupby("id")
            .sum()
            .drop(index=[-1])
        )

        ax0.fill_between(out.index, cumprod_capital, cumprod_close, where=greater_eq, color='red', alpha=0.2,
                         edgecolor='red')
        ax0.fill_between(out.index, cumprod_capital, cumprod_close, where=less, color='green', alpha=0.2,
                         edgecolor='green')

        ax0.axvline(x=self.out.index[first_trade], color='grey', linestyle='--', alpha=0.5, label="First trade")

        ax0.legend()

        ax1.plot(out['close'], label="close")

        ax1.plot(out.loc[:, ["close"]][out.start == 1].dropna().index.values,
                 out.loc[:, ["close"]][out.start == 1].dropna().values, marker=11, linestyle="None",
                 markersize=10, color='g', label="Enter trade")
        ax1.plot(out.loc[:, ["close"]][out.end == 1].dropna().index.values,
                 out.loc[:, ["close"]][out.end == 1].dropna().values, marker=11, linestyle="None", color='r',
                 markersize=10, label="Exit trade")

        ax1.plot(out.loc[:, ["close"]][(out.end == 1) & (out.start == 1)].dropna().index.values,
                 out.loc[:, ["close"]][(out.end == 1) & (out.start == 1)].dropna().values, marker=11,
                 linestyle="None", color='yellow', markersize=10, label="Inverse trade")

        for indicator in indicators_to_plot:
            if self.many:
                ax1.plot(self.indicators[instrument][indicator], label=indicator)
            else:
                ax1.plot(self.indicators[indicator], label=indicator)

        ax1.legend()

        ax2.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        colors = ['green' if val > 0 else 'red' for val in sum_returns.values]

        if (len(np.where(out.end == 1)[0] != len(sum_returns))) and (out.id[out.index[-1]] != -1) \
                and (out.start[out.index[-1]] != 1):
            out.loc[out.index[-1], ["end"]] = 1

        ax2.scatter(out.index[np.where(out.end == 1)[0]], sum_returns, c=colors, s=100)

        ax0.set_xlim([out.index[0], out.index[-1]])
        ax1.set_xlim([out.index[0], out.index[-1]])
        ax2.set_xlim([out.index[0], out.index[-1]])
        plt.show()

    @staticmethod
    def append_array_to_dataframe(df, arr, new_col_name='new_col'):
        """
        Dodajemy arraya na koniec dataframe
        :param new_col_name:
        :type new_col_name:
        :param df:
        :type df:
        :param arr:
        :type arr:
        :return:
        :rtype:
        """
        df.loc[:, new_col_name] = np.nan
        df.iloc[-len(arr):, -1] = arr
        return df

    def combine_results(self, close, orders):
        """
        Przetwarzamy brzydkiego Cythonowego arraya na pd.DataFrame
        """
        columns = ("id", "direction", "capital", "returns", "sl/tp", "start", "end")

        for i in range(0, len(columns)):
            close = self.append_array_to_dataframe(close, orders[:, i], columns[i])

        return close
