# CyTester
___
CyTester is a python package that allows you to backtest your trading strategy in a simple and intuitive manner. 

### Why should you use it?

It is event based and has been designed to let the user access any variable at the moment _t_, while still being fast because of its Cython based engine.

### Is CyTester a finished project?

**No.** CyTester is still being developed and optimized. New features will be added regularly.

### What can CyTester do at the moment?

1. It can backtest any strategy based on a pandas DataFrame.
2. It can backtest multiple instruments at the same time **and do it quickly** because of multiprocessing.
3. It can take various parameters that will adjust it to your needs.
4. It can provide you with basic visualizations of your strategy and the indicators.
5. It can provide you with basic statistics to evaluate your strategy.

Let's make an artificial dataset:

`close = (pd.DataFrame({"close": np.array(np.random.normal(0, 1, 100))}).cumsum() + 100).abs()`

And let's create a moving average to it:

```
indicators = (
    close
    .assign(ma5=lambda x: x.rolling(5).mean())
    .drop(columns='close')
)
```

The decision function should be defined like:

```angular2html
def decision(close, indicators, i, previous_state):
    if (indicators[i][0] < close[i]) and (indicators[i - 1][0] > close[i - 1]):
        return 1
    elif indicators[i][0] > close[i] and indicators[i - 1][0] < close[i - 1]:
        return 0
    else:
        return previous_state
```

This function must return:

a) `1` - enter long trade

b) `-1` - enter short trade

c) `0` - exit trade

d) `previous_state` - continue previous trade


Now let's create a CyTester instance and run the backtest:

```angular2html
backtest = CyTester(data=close,
                    decision=decision,
                    indicators=indicators,
                    starting_index=1,
                    leverage=1,
                    one_lot_worth=1,
                    lot=1,
                    static_sl=10,
                    const_spread=1)
backtest.run()
```