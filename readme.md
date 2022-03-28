# Intro:

A DQN-based trading strategy applying different reward types.

Current type of rewards:

- Simple daily return
- Sortino ratio
- Sharpe ratio

# Run steps:

1. ```mkdir DQN-Forex```

2. ```git clone https://github.com/brunorabelo/DQN-Forex.git```
3. ```cd DQN-Forex```
4. ```virtualenv venv```
5. ```source venv/bin/activate```
6. ```pip install -r requirements.txt```

### run code:

```python3 main.py```


# Description:
It is still in development.

# Future work:
- [ ] Add more reward types
- [ ] Use more data
- [ ] Use different window times
- [ ] Use a set for testing and apply backtesting
