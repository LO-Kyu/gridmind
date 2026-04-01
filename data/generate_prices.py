"""Generate 30 days of realistic ISO New England-style hourly price data."""
import json
import math
import random

random.seed(12345)

def generate_day(day_type: str = "weekday") -> list[float]:
    """Generate 24 hourly prices for one day."""
    prices = []
    for hour in range(24):
        # Base time-of-use price
        if hour < 6:  # overnight off-peak
            base = 0.035 + random.gauss(0, 0.005)
        elif hour < 8:  # morning ramp
            base = 0.07 + random.gauss(0, 0.008)
        elif hour < 12:  # morning peak
            base = 0.16 + random.gauss(0, 0.020) + (0.05 if day_type == "weekday" else 0)
        elif hour < 14:  # midday moderate
            base = 0.10 + random.gauss(0, 0.012)
        elif hour < 17:  # afternoon
            base = 0.12 + random.gauss(0, 0.015)
        elif hour < 21:  # evening super-peak
            base = 0.22 + random.gauss(0, 0.025) + (0.08 if day_type == "weekday" else 0.02)
        elif hour < 23:  # evening wind-down
            base = 0.09 + random.gauss(0, 0.010)
        else:  # late night
            base = 0.04 + random.gauss(0, 0.006)

        # Seasonal variation (summer = higher peaks)
        season_mult = 1.0
        prices.append(round(max(0.02, base * season_mult), 4))
    return prices


# Generate 30 days: 22 weekdays + 8 weekend days
days = []
day_types = (["weekday"] * 5 + ["weekend"] * 2) * 5  # 5 weeks → 35 days → take 30
for i, dt in enumerate(day_types[:30]):
    day_data = {
        "day": i + 1,
        "day_type": dt,
        "prices_usd_per_kwh": generate_day(dt),
        "source": "synthetic_iso_new_england_style",
        "region": "ISONE",
        "currency": "USD",
    }
    days.append(day_data)

output = {
    "description": "30 days of synthetic ISO New England-style hourly electricity price data",
    "units": "USD/kWh",
    "hours_per_day": 24,
    "days": days,
}

with open("price_curves.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(days)} days of price data -> data/price_curves.json")
