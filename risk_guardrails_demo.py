
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class ValidationDecision(Enum):

    APPROVED = "APPROVED"
    CONDITIONAL = "CONDITIONAL"
    REJECTED = "REJECTED"


class V19RiskGuardrails:

    def __init__(self):

        self.max_confidence = 0.42

        self.thresholds = {
            'min_confidence': 0.15,
            'min_validation_score': 0.70,
            'max_risk_score': 0.80,
            'high_confidence': 0.25,
            'high_validation': 0.80
        }

    def validate_prediction(self, market_data: Dict, prediction: Dict) -> Dict:

        print("\n" + "=" * 60)
        print("V19.4 Five layer Test System")
        print("=" * 60)

        print("\n[Layer 1] Data Test...")
        data_score, data_issues = self.layer1_data_validation(market_data)
        print(f"  Data Score: {data_score:.2f}")
        if data_issues:
            print(f"  Remark: {', '.join(data_issues)}")

        print("\n[Layer 2] Risk Evaluation...")
        risk_score, risk_factors = self.layer2_risk_assessment(market_data)
        print(f"  Risk Score: {risk_score:.2f}")
        if risk_factors:
            print(f"  Risk Factor: {', '.join(risk_factors)}")

        print("\n[Layer 3] Exception Check...")
        anomaly_score, anomalies = self.layer3_anomaly_detection(market_data)
        print(f"  Exception Score: {anomaly_score:.2f}")
        if anomalies:
            print(f"  Exception: {', '.join(anomalies)}")

        print("\n[Layer 4] Consistency Score...")
        consistency_score, consistency_issues = self.layer4_consistency_check(prediction)
        print(f"  Consistency Score: {consistency_score:.2f}")
        if consistency_issues:
            print(f"  Consistency Problem: {', '.join(consistency_issues)}")

        print("\n[Layer 5] Decision...")
        decision, reason, overall_score = self.layer5_final_decision(
            prediction, data_score, risk_score, anomaly_score, consistency_score
        )
        print(f"  Integrated Score: {overall_score:.2f}")
        print(f"  Decision Result: {decision.value}")
        print(f"  Decision Reason: {reason}")

        return {
            'decision': decision,
            'reason': reason,
            'overall_score': overall_score,
            'layer_scores': {
                'data': data_score,
                'risk': risk_score,
                'anomaly': anomaly_score,
                'consistency': consistency_score
            },
            'issues': {
                'data': data_issues,
                'risk': risk_factors,
                'anomaly': anomalies,
                'consistency': consistency_issues
            }
        }

    def layer1_data_validation(self, data: Dict) -> Tuple[float, list]:
        issues = []
        score = 1.0

        required_fields = ['price', 'price_change_24h', 'fear_greed_index',
                          'open_interest', 'liquidation_24h']
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(f"lack_fields: {field}")
                score -= 0.2

        if 'price' in data:
            if data['price'] <= 0 or data['price'] > 100000:
                issues.append("Price Alert")
                score -= 0.3

        if 'fear_greed_index' in data:
            if not (0 <= data['fear_greed_index'] <= 100):
                issues.append("Fear Greed Index Alert")
                score -= 0.2

        if 'price_change_24h' in data:
            if abs(data['price_change_24h']) > 50:
                issues.append("Price change rapidly")
                score -= 0.1

        return max(0, score), issues

    def layer2_risk_assessment(self, data: Dict) -> Tuple[float, list]:

        risk_factors = []
        risk_score = 0.0

        fear_greed = data.get('fear_greed_index', 50)
        if fear_greed > 70:
            risk_factors.append("Over-Greedy")
            risk_score += 0.3
        elif fear_greed < 30:
            risk_factors.append("Over-Fear")
            risk_score += 0.2

        price_change = data.get('price_change_24h', 0)
        if abs(price_change) > 5:
            risk_factors.append("Price change rapidly")
            risk_score += 0.2

        liquidation = data.get('liquidation_24h', 0)
        if liquidation > 200e6:
            risk_factors.append("High liquidation pressure")
            risk_score += 0.3

        oi = data.get('open_interest', 50e9)
        if oi > 70e9:
            risk_factors.append("Open interest too high")
            risk_score += 0.2

        return min(1.0, risk_score), risk_factors

    def layer3_anomaly_detection(self, data: Dict) -> Tuple[float, list]:
        anomalies = []
        anomaly_score = 0.0

        price_change = data.get('price_change_24h', 0)
        if abs(price_change) > 10:
            anomalies.append("Extreme Price Change")
            anomaly_score += 0.4

        volume = data.get('volume_24h', 1e9)
        if volume > 5e9 or volume < 0.5e9:
            anomalies.append("Abnormal Volume")
            anomaly_score += 0.2

        liquidation = data.get('liquidation_24h', 0)
        if liquidation > 500e6:
            anomalies.append("Extreme Liquidation")
            anomaly_score += 0.4

        fear_greed = data.get('fear_greed_index', 50)
        if (fear_greed > 70 and price_change < -3) or (fear_greed < 30 and price_change > 3):
            anomalies.append("special fear-greed-index and price relationship ")
            anomaly_score += 0.3

        return min(1.0, anomaly_score), anomalies

    def layer4_consistency_check(self, prediction: Dict) -> Tuple[float, list]:

        issues = []
        score = 1.0

        probabilities = prediction.get('probabilities', [])
        if len(probabilities) != 5:
            issues.append("Distribution Wrong")
            score -= 0.5
        elif abs(sum(probabilities) - 1.0) > 0.01:
            issues.append("Total Probabilties not equal to 1")
            score -= 0.3

        confidence = prediction.get('confidence', 0)
        if confidence > self.max_confidence:
            issues.append(f"confidence exceeds limit({self.max_confidence})")
            score -= 0.2

        direction = prediction.get('direction', '')
        if probabilities:
            max_prob_idx = np.argmax(probabilities)
            expected_directions = ['Extreme Short(>3%)', 'Short(1-3%)', 'Oscillation(±1%)', 'Long(1-3%)', 'Extreme Long(>3%)']
            if max_prob_idx < len(expected_directions):
                expected_direction = expected_directions[max_prob_idx]
                if direction != expected_direction:
                    issues.append("Direction and Probabilities not consistent")
                    score -= 0.4

        return max(0, score), issues

    def layer5_final_decision(self, prediction: Dict, data_score: float,
                             risk_score: float, anomaly_score: float,
                             consistency_score: float) -> Tuple[ValidationDecision, str, float]:
        confidence = prediction.get('confidence', 0)

        overall_score = (
            data_score * 0.3 +
            consistency_score * 0.3 +
            (1 - risk_score) * 0.2 +
            (1 - anomaly_score) * 0.2
        )

        if confidence < self.thresholds['min_confidence']:
            return ValidationDecision.REJECTED, "Confidence too Low", overall_score

        if overall_score < self.thresholds['min_validation_score']:
            return ValidationDecision.REJECTED, "Score insufficient", overall_score

        if risk_score > self.thresholds['max_risk_score']:
            return ValidationDecision.REJECTED, "Risk too high", overall_score

        if anomaly_score > 0.7:
            return ValidationDecision.REJECTED, "Serious Exception", overall_score

        if (confidence >= self.thresholds['high_confidence'] and
            overall_score >= self.thresholds['high_validation']):
            return ValidationDecision.APPROVED, "Approved, trade confidently", overall_score

        if (confidence >= 0.20 and overall_score >= 0.70):
            return ValidationDecision.CONDITIONAL, "Conditional，carefully trade", overall_score

        return ValidationDecision.CONDITIONAL, "Conditional, conservative trade", overall_score

    def limit_confidence(self, raw_confidence: float) -> float:

        return min(raw_confidence, self.max_confidence)

    def calculate_position_size(self, base_size: float, confidence: float,
                               risk_score: float) -> float:

        confidence_factor = confidence / self.max_confidence
        risk_factor = 1 - risk_score

        adjusted_size = base_size * confidence_factor * risk_factor

        max_size = 0.20
        min_size = 0.05

        return max(min_size, min(adjusted_size, max_size))


def demo():

    print("=" * 60)
    print("ETH V19 Risk Guardrails Demo")
    print("=" * 60)

    guardrails = V19RiskGuardrails()

    print("\n\n【Test case 1: Narmal market Condition】")
    market_data_1 = {
        'price': 4505.87,
        'price_change_24h': -1.65,
        'volume_24h': 3.23e9,
        'fear_greed_index': 51,
        'open_interest': 61.46e9,
        'liquidation_24h': 99.73e6
    }

    prediction_1 = {
        'direction': 'Extreme Short(>3%)',
        'confidence': 0.42,
        'probabilities': [0.42, 0.28, 0.18, 0.08, 0.04]
    }

    result_1 = guardrails.validate_prediction(market_data_1, prediction_1)

    print("\n\n【Test case 2: High Risk COndition (V19.3failure case)】")
    market_data_2 = {
        'price': 3684.37,
        'price_change_24h': 3.23,
        'volume_24h': 2.5e9,
        'fear_greed_index': 71,
        'open_interest': 55.34e9,
        'liquidation_24h': 166.44e6
    }

    prediction_2 = {
        'direction': 'Extreme Long(>3%)',
        'confidence': 0.519,
        'probabilities': [0.016, 0.049, 0.097, 0.319, 0.519]
    }

    prediction_2['confidence'] = guardrails.limit_confidence(prediction_2['confidence'])
    print(f"\n Apply Confidence Limit: 0.519 → {prediction_2['confidence']}")

    result_2 = guardrails.validate_prediction(market_data_2, prediction_2)

    print("\n\n【test Case 3: Extreme Condition】")
    market_data_3 = {
        'price': 4200.0,
        'price_change_24h': -15.5,
        'volume_24h': 8.5e9,
        'fear_greed_index': 15,
        'open_interest': 75e9,
        'liquidation_24h': 600e6
    }

    prediction_3 = {
        'direction': 'Extreme Short(>3%)',
        'confidence': 0.35,
        'probabilities': [0.50, 0.30, 0.10, 0.07, 0.03]
    }

    result_3 = guardrails.validate_prediction(market_data_3, prediction_3)

    print("\n\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nCase 1 (Normal Condition):")
    print(f"  Decision: {result_1['decision'].value}")
    print(f"  Integrated Score: {result_1['overall_score']:.2f}")

    print("\nCase2 (V19.3 Failure Case):")
    print(f"  Decision: {result_2['decision'].value}")
    print(f"  Integrated Score: {result_2['overall_score']:.2f}")
    print(f"  V19.4Recovery: Confidence Limit + Greedy Risk Detection")

    print("\nCAse3 (Extreme Condition):")
    print(f"  Decision: {result_3['decision'].value}")
    print(f"  Integrated Score: {result_3['overall_score']:.2f}")

    print("\n\n" + "=" * 60)
    print("Dynamic Position Size Sample")
    print("=" * 60)

    base_size = 0.20

    for i, (result, label) in enumerate([(result_1, "Case1"), (result_2, "Case2"), (result_3, "case3")], 1):
        confidence = [prediction_1, prediction_2, prediction_3][i-1]['confidence']
        risk_score = result['layer_scores']['risk']

        position = guardrails.calculate_position_size(base_size, confidence, risk_score)
        print(f"\n{label}:")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Risk Score: {risk_score:.2f}")
        print(f"  Position Size Reccomendation: {position*100:.1f}%")

    print("\n" + "=" * 60)
    print("Risk_guardrails complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
