@app.get("/nifty")
def nifty():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"
    }

    try:
        session = requests.Session()

        # NSE से पहले homepage खोलकर session/cookies लेना
        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=10
        )

        response = session.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return {
                "symbol": "NIFTY 50",
                "price": None,
                "status": "nse_error",
                "http_status": response.status_code
            }

        data = response.json()
        info = data["metadata"]

        return {
            "symbol": "NIFTY 50",
            "price": info.get("last"),
            "change": info.get("change"),
            "change_percent": info.get("percentChange"),
            "market_status": "live",
            "time": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
        info = data["metadata"]

        return {
            "symbol": "NIFTY 50",
            "price": info.get("last"),
            "change": info.get("change"),
            "change_percent": info.get("percentChange"),
            "market_status": "live",
            "time": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
