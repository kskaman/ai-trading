"""
Trading strategies.

Each strategy turns a close-price panel (src.market_data.panel) into target weights
(src.strategies.weights) and, through BaseStrategy.generate_signals, into
entry/exit events.

    from src.strategies.momentum import CrossSectionalMomentum
    from src.strategies.pairs_trading import PairsTradingStrategy, check_cointegration
    from src.strategies.kalman_pairs import KalmanPairsStrategy
    from src.strategies.factors import MultiFactorModel
"""
