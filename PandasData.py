import backtrader as bt

class PandasData(bt.feed.DataBase):
    lines = ('time', 'open', 'close', 'high', 'low', 'volume')
    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('time', 'time'),
        ('open', 1),
        ('close', 2),
        ('high', 3),
        ('low', 4),
        ('volume', 5)
    )