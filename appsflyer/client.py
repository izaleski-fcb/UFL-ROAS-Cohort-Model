import os
import io
import requests
import pandas as pd
from dotenv import load_dotenv

# В облачном окружении .env не будет — load_dotenv на несуществующий путь просто
# ничего не делает, и os.getenv возьмёт значения из реальных env vars (секретов),
# заданных в конфигурации окружения claude.ai.
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

AF_API_TOKEN = os.getenv('AF_API_TOKEN')
AF_APP_ID_IOS = os.getenv('AF_APP_ID_IOS')
AF_APP_ID_ANDROID = os.getenv('AF_APP_ID_ANDROID')

BASE_URL = "https://hq1.appsflyer.com/api"


class AppsFlyerClient:
    def __init__(self, app_id: str):
        self.app_id = app_id
        self.token = AF_API_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        })

    def get_cohort_revenue(self, date_from: str, date_to: str, groupings: list = None) -> pd.DataFrame:
        """
        Когортный отчёт по revenue (IAP + Ad Revenue).
        date_from / date_to: формат YYYY-MM-DD
        groupings: список полей группировки, например ['pid'] (media source) или ['pid', 'c']
        """
        if groupings is None:
            groupings = ["pid", "c"]

        url = f"{BASE_URL}/cohorts/v1/data/app/{self.app_id}"

        payload = {
            "cohort_type": "user_acquisition",
            "min_cohort_size": 1,
            "aggregation_type": "cumulative",
            "groupings": groupings,
            "kpis": ["revenue"],
            "from": date_from,
            "to": date_to,
            "preferred_currency": True,
            "cohort_periods": ["day_1", "day_3", "day_7", "day_14", "day_30", "day_60", "day_90"]
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        if not response.text.strip():
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(response.text))


def get_ios_client() -> AppsFlyerClient:
    return AppsFlyerClient(app_id=AF_APP_ID_IOS)


def get_android_client() -> AppsFlyerClient:
    return AppsFlyerClient(app_id=AF_APP_ID_ANDROID)
