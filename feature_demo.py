import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class V19FeatureEngine:
    """V19 feature_list"""

    def __init__(self):
        self.feature_count = 80
        self.feature_categories = {
            'price': 10,
            'volume': 8,
            'technical': 12,
            'sentiment': 8,
            'derivatives': 15,
            'onchain': 8,
            'cross': 19
        }

    def extract_all_features(self, market_data: Dict) -> Dict[str, float]:

        features = {}

        features.update(self.extract_price_features(market_data))

        features.update(self.extract_volume_features(market_data))

        features.update(self.extract_technical_features(market_data))

        features.update(self.extract_sentiment_features(market_data))

        features.update(self.extract_derivatives_features(market_data))

        features.update(self.extract_onchain_features(market_data))

        features.update(self.extract_cross_features(features, market_data))

        assert len(features) == 80, f"Feature count mismatch: {len(features)} != 80"
        return features

    def extract_price_features(self, data: Dict) -> Dict[str, float]:

        price = data['price']
        price_change_24h = data.get('price_change_24h', 0)
        high_24h = data.get('high_24h', price * 1.05)
        low_24h = data.get('low_24h', price * 0.95)

        price_normalized = price / 4000.0

        price_change_rate = price_change_24h / 100.0

        price_change_1h = price_change_rate * 0.05
        price_change_4h = price_change_rate * 0.20

        price_position_high = (price - low_24h) / (high_24h - low_24h + 1e-6)
        price_position_low = (high_24h - price) / (high_24h - low_24h + 1e-6)

        price_volatility = (high_24h - low_24h) / price

        price_momentum = abs(price_change_rate) * np.sign(price_change_rate)

        price_trend_strength = abs(price_change_rate)

        price_mid = (high_24h + low_24h) / 2
        price_deviation = (price - price_mid) / price_mid

        return {
            'price_normalized': price_normalized,
            'price_change_24h': price_change_rate,
            'price_change_1h': price_change_1h,
            'price_change_4h': price_change_4h,
            'price_position_high': price_position_high,
            'price_position_low': price_position_low,
            'price_volatility': price_volatility,
            'price_momentum': price_momentum,
            'price_trend_strength': price_trend_strength,
            'price_deviation': price_deviation
        }

    def extract_volume_features(self, data: Dict) -> Dict[str, float]:

        volume_24h = data.get('volume_24h', 1e9)
        price_change = data.get('price_change_24h', 0)

        volume_normalized = volume_24h / 1e10

        volume_change_rate = 0.1 if volume_24h > 1.2e10 else -0.1

        volume_trend = 1.0 if volume_change_rate > 0 else -1.0

        volume_volatility = abs(volume_change_rate)

        price_volume_corr = np.sign(price_change) * np.sign(volume_change_rate)

        volume_anomaly = 1.0 if volume_24h > 1.5e10 else 0.0

        volume_concentration = min(volume_24h / 2e10, 1.0)

        volume_momentum = volume_change_rate * volume_trend

        return {
            'volume_normalized': volume_normalized,
            'volume_change_rate': volume_change_rate,
            'volume_trend': volume_trend,
            'volume_volatility': volume_volatility,
            'price_volume_corr': price_volume_corr,
            'volume_anomaly': volume_anomaly,
            'volume_concentration': volume_concentration,
            'volume_momentum': volume_momentum
        }

    def extract_technical_features(self, data: Dict) -> Dict[str, float]:

        price = data['price']
        price_change = data.get('price_change_24h', 0)

        rsi = 50 + price_change * 2
        rsi = max(0, min(100, rsi))
        rsi_normalized = rsi / 100.0

        macd_signal = np.tanh(price_change / 10.0)

        bollinger_position = 0.5 + price_change / 20.0
        bollinger_position = max(0, min(1, bollinger_position))

        ma5_deviation = price_change / 5.0
        ma20_deviation = price_change / 20.0

        ma_trend = 1.0 if price_change > 0 else -1.0

        support_distance = abs(min(price_change, 0)) / 5.0
        resistance_distance = abs(max(price_change, 0)) / 5.0

        breakout_signal = 1.0 if abs(price_change) > 3.0 else 0.0

        overbought_signal = 1.0 if rsi > 70 else 0.0
        oversold_signal = 1.0 if rsi < 30 else 0.0
        overbought_oversold = overbought_signal - oversold_signal

        trend_strength = abs(price_change) / 10.0

        reversal_signal = 1.0 if (rsi > 70 and price_change > 0) or (rsi < 30 and price_change < 0) else 0.0

        return {
            'rsi_normalized': rsi_normalized,
            'macd_signal': macd_signal,
            'bollinger_position': bollinger_position,
            'ma5_deviation': ma5_deviation,
            'ma20_deviation': ma20_deviation,
            'ma_trend': ma_trend,
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'breakout_signal': breakout_signal,
            'overbought_oversold': overbought_oversold,
            'trend_strength': trend_strength,
            'reversal_signal': reversal_signal
        }

    def extract_sentiment_features(self, data: Dict) -> Dict[str, float]:

        fear_greed_index = data.get('fear_greed_index', 50)
        price_change = data.get('price_change_24h', 0)

        fear_signal = (100 - fear_greed_index) / 100.0

        sentiment_extreme = abs(fear_greed_index - 50) / 50.0

        sentiment_change_rate = (fear_greed_index - 50) / 100.0

        sentiment_price_divergence = np.sign(fear_greed_index - 50) * np.sign(-price_change)

        panic_signal = 1.0 if fear_greed_index < 30 else 0.0

        greed_signal = 1.0 if fear_greed_index > 70 else 0.0

        sentiment_volatility = sentiment_extreme

        sentiment_trend = 1.0 if fear_greed_index > 50 else -1.0

        return {
            'fear_signal': fear_signal,
            'sentiment_extreme': sentiment_extreme,
            'sentiment_change_rate': sentiment_change_rate,
            'sentiment_price_divergence': sentiment_price_divergence,
            'panic_signal': panic_signal,
            'greed_signal': greed_signal,
            'sentiment_volatility': sentiment_volatility,
            'sentiment_trend': sentiment_trend
        }

    def extract_derivatives_features(self, data: Dict) -> Dict[str, float]:

        open_interest = data.get('open_interest', 50e9)
        liquidation_24h = data.get('liquidation_24h', 100e6)
        price_change = data.get('price_change_24h', 0)

        oi_normalized = open_interest / 60e9

        oi_change_rate = (open_interest - 55e9) / 55e9

        liquidation_normalized = liquidation_24h / 200e6

        long_liquidation_ratio = 0.7 if price_change < 0 else 0.3
        short_liquidation_ratio = 1.0 - long_liquidation_ratio

        liquidation_imbalance = long_liquidation_ratio - short_liquidation_ratio

        funding_rate = price_change / 100.0

        long_short_ratio = 1.2 if price_change > 0 else 0.8

        futures_premium = price_change / 50.0

        option_skew = -price_change / 100.0

        derivatives_volume = liquidation_normalized * 10

        leverage_usage = (liquidation_normalized + oi_normalized) / 2

        liquidation_risk = liquidation_normalized * abs(liquidation_imbalance)

        cascade_risk = liquidation_risk * leverage_usage

        derivatives_sentiment = np.tanh(funding_rate + futures_premium)

        return {
            'oi_normalized': oi_normalized,
            'oi_change_rate': oi_change_rate,
            'liquidation_normalized': liquidation_normalized,
            'long_liquidation_ratio': long_liquidation_ratio,
            'short_liquidation_ratio': short_liquidation_ratio,
            'liquidation_imbalance': liquidation_imbalance,
            'funding_rate': funding_rate,
            'long_short_ratio': long_short_ratio,
            'futures_premium': futures_premium,
            'option_skew': option_skew,
            'derivatives_volume': derivatives_volume,
            'leverage_usage': leverage_usage,
            'liquidation_risk': liquidation_risk,
            'cascade_risk': cascade_risk,
            'derivatives_sentiment': derivatives_sentiment
        }

    def extract_onchain_features(self, data: Dict) -> Dict[str, float]:

        price_change = data.get('price_change_24h', 0)
        fear_greed = data.get('fear_greed_index', 50)
        staking_exits = data.get('staking_exits', 100000)

        base_exit = 100000
        price_factor = max(0, price_change) * 10000
        fear_factor = max(0, fear_greed - 60) * 2000
        staking_exit_estimate = (base_exit + price_factor + fear_factor) / 1e6

        staking_inflow_estimate = max(0, -price_change) * 5000 / 1e6

        staking_net_flow = staking_exit_estimate - staking_inflow_estimate

        active_addresses = 0.5 + abs(price_change) / 20.0

        whale_movement = abs(price_change) / 10.0

        exchange_net_flow = -price_change / 10.0

        onchain_volume = 0.5 + abs(price_change) / 10.0

        network_activity = (active_addresses + onchain_volume) / 2

        return {
            'staking_exit_estimate': staking_exit_estimate,
            'staking_inflow_estimate': staking_inflow_estimate,
            'staking_net_flow': staking_net_flow,
            'active_addresses': active_addresses,
            'whale_movement': whale_movement,
            'exchange_net_flow': exchange_net_flow,
            'onchain_volume': onchain_volume,
            'network_activity': network_activity
        }

    def extract_cross_features(self, features: Dict, data: Dict) -> Dict[str, float]:

        price_sentiment_cross = features['price_change_24h'] * features['fear_signal']

        price_volume_cross = features['price_change_24h'] * features['volume_normalized']

        price_derivatives_cross = features['price_change_24h'] * features['liquidation_normalized']

        sentiment_derivatives_cross = features['fear_signal'] * features['liquidation_risk']

        technical_sentiment_cross = features['rsi_normalized'] * features['fear_signal']

        volume_derivatives_cross = features['volume_normalized'] * features['oi_normalized']

        onchain_price_cross = features['staking_net_flow'] * features['price_change_24h']

        risk_cross_1 = features['liquidation_risk'] * features['greed_signal']
        risk_cross_2 = features['cascade_risk'] * features['sentiment_extreme']
        risk_cross_3 = features['staking_exit_estimate'] * features['fear_signal']

        trend_consistency_1 = features['price_trend_strength'] * features['ma_trend']
        trend_consistency_2 = features['volume_trend'] * features['sentiment_trend']

        volatility_cross_1 = features['price_volatility'] * features['volume_volatility']
        volatility_cross_2 = features['sentiment_volatility'] * features['liquidation_normalized']

        momentum_cross_1 = features['price_momentum'] * features['volume_momentum']
        momentum_cross_2 = features['price_momentum'] * features['derivatives_sentiment']

        extreme_cross_1 = features['breakout_signal'] * features['volume_anomaly']
        extreme_cross_2 = features['reversal_signal'] * features['sentiment_extreme']

        comprehensive_risk = (features['liquidation_risk'] + features['greed_signal'] +
                  features['staking_exit_estimate']) / 3

        return {
            'price_sentiment_cross': price_sentiment_cross,
            'price_volume_cross': price_volume_cross,
            'price_derivatives_cross': price_derivatives_cross,
            'sentiment_derivatives_cross': sentiment_derivatives_cross,
            'technical_sentiment_cross': technical_sentiment_cross,
            'volume_derivatives_cross': volume_derivatives_cross,
            'onchain_price_cross': onchain_price_cross,
            'risk_cross_1': risk_cross_1,
            'risk_cross_2': risk_cross_2,
            'risk_cross_3': risk_cross_3,
            'trend_consistency_1': trend_consistency_1,
            'trend_consistency_2': trend_consistency_2,
            'volatility_cross_1': volatility_cross_1,
            'volatility_cross_2': volatility_cross_2,
            'momentum_cross_1': momentum_cross_1,
            'momentum_cross_2': momentum_cross_2,
            'extreme_cross_1': extreme_cross_1,
            'extreme_cross_2': extreme_cross_2,
            'comprehensive_risk': comprehensive_risk
        }


def demo():

    print("=" * 60)
    print("ETH V19 Feature Engineering Demo")
    print("=" * 60)

    market_data = {
        'price': 4505.87,
        'price_change_24h': -1.65,
        'high_24h': 4600.0,
        'low_24h': 4450.0,
        'volume_24h': 3.23e9,
        'fear_greed_index': 51,
        'open_interest': 61.46e9,
        'liquidation_24h': 99.73e6,
        'staking_exits': 150000
    }

    print("\n Input Data:")
    for key, value in market_data.items():
        print(f"  {key}: {value}")

    engine = V19FeatureEngine()

    print("\nGetting 80 features...")
    features = engine.extract_all_features(market_data)

    print(f"\n Getting {len(features)} features")

    print("\nFeature Stats:")
    categories = {
        'price': [k for k in features.keys() if k.startswith('price_')],
        'volume': [k for k in features.keys() if k.startswith('volume_')],
        'technical': [k for k in features.keys() if any(x in k for x in ['rsi', 'macd', 'bollinger', 'ma_', 'support', 'resistance', 'breakout', 'overbought', 'trend_strength', 'reversal'])],
        'sentiment': [k for k in features.keys() if any(x in k for x in ['fear', 'sentiment', 'panic', 'greed'])],
        'derivatives': [k for k in features.keys() if any(x in k for x in ['oi_', 'liquidation', 'funding', 'long_', 'short_', 'futures', 'option', 'derivatives', 'leverage', 'cascade'])],
        'onchain': [k for k in features.keys() if any(x in k for x in ['staking', 'active_addresses', 'whale', 'exchange', 'onchain', 'network'])],
        'cross': [k for k in features.keys() if 'cross' in k or k == 'comprehensive_risk']
    }

    for category, feature_list in categories.items():
        print(f"  {category}: {len(feature_list)} features")

    print("\n V19.4 Key Features:")
    print(f"  fear_signal: {features['fear_signal']:.4f}")
    print(f"  greed_signal: {features['greed_signal']:.4f}")
    print(f"  staking_exit_estimate: {features['staking_exit_estimate']:.4f}")

    print("\n Risk Feature:")
    print(f"  liquidation_risk: {features['liquidation_risk']:.4f}")
    print(f"  cascade_risk: {features['cascade_risk']:.4f}")
    print(f"  comprehensive_risk: {features['comprehensive_risk']:.4f}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
