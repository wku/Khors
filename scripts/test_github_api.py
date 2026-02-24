import os
import requests
import json
from dotenv import load_dotenv

def test_github_access():
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN не найден в окружении")
        return

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Khors-Test-Script"
    }
    
    # 1. Проверка токена (кто я?)
    user_url = "https://api.github.com/user"
    r = requests.get(user_url, headers=headers)
    if r.status_code == 200:
        user_data = r.json()
        print(f"✅ Токен валиден. Пользователь: {user_data.get('login')}")
        print(f"Scopes: {r.headers.get('X-OAuth-Scopes')}")
    else:
        print(f"❌ Ошибка проверки токена: {r.status_code}")
        print(r.text)
        return

    # 2. Проверка доступа к репозиторию
    repo = "wku/Khors"
    repo_url = f"https://api.github.com/repos/{repo}"
    r = requests.get(repo_url, headers=headers)
    if r.status_code == 200:
        print(f"✅ Доступ к репозиторию {repo} подтвержден")
    else:
        print(f"❌ Нет доступа к репозиторию {repo}: {r.status_code}")
        print(r.text)

if __name__ == "__main__":
    test_github_access()
